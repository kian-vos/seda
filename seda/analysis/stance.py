"""Stance classification for SEDA."""

from typing import Optional

from seda.config import get_settings
from seda.db import get_db
from seda.models import (
    Account,
    AccountType,
    PoliticalStance,
    PoliticalTaxonomy,
    ThreatLevel,
    Tweet,
)
from seda.analysis.nlp import get_nlp

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class StanceClassifier:
    """Classify political stance of accounts."""

    def __init__(self):
        self.db = get_db()
        self.nlp = get_nlp()
        self.settings = get_settings()
        self.client = None

        # Initialize Anthropic client if available
        if ANTHROPIC_AVAILABLE and self.settings.anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    def classify_account(
        self,
        account: Account,
        tweets: Optional[list[Tweet]] = None,
        use_llm: bool = True,
    ) -> tuple[PoliticalStance, Optional[PoliticalTaxonomy], ThreatLevel]:
        """Classify an account's political stance and threat level.

        Args:
            account: Account to classify
            tweets: Account's tweets (fetched if not provided)
            use_llm: Whether to use LLM for nuanced classification

        Returns:
            Tuple of (stance, taxonomy, threat_level)
        """
        if tweets is None and account.id:
            tweets = self.db.get_tweets_by_account(account.id, limit=50)

        # First try rule-based classification
        stance, taxonomy = self._rule_based_classify(account, tweets or [])

        # If neutral or unknown and LLM available, try LLM classification
        if (
            stance in [PoliticalStance.NEUTRAL, PoliticalStance.UNKNOWN]
            and use_llm
            and self.client
            and tweets
        ):
            llm_stance, llm_taxonomy = self._llm_classify(account, tweets)
            if llm_stance != PoliticalStance.UNKNOWN:
                stance = llm_stance
                taxonomy = llm_taxonomy

        # Create a temporary account copy with the stance for threat classification
        account_with_stance = Account(**account.model_dump())
        account_with_stance.political_stance = stance

        # Classify threat level for pro-regime accounts
        threat_level = self.classify_threat_level(account_with_stance, tweets or [])

        return stance, taxonomy, threat_level

    def _rule_based_classify(
        self, account: Account, tweets: list[Tweet]
    ) -> tuple[PoliticalStance, Optional[PoliticalTaxonomy]]:
        """Classify using keyword rules and network signals."""
        pro_score = 0.0
        anti_score = 0.0

        # Seed accounts are by definition pro-regime
        if account.is_seed:
            taxonomy = self._determine_seed_taxonomy(account)
            return PoliticalStance.PRO_REGIME, taxonomy

        # Check bio for keywords
        if account.bio:
            bio_pro = len(self.nlp.detect_pro_regime_keywords(account.bio))
            bio_anti = len(self.nlp.detect_opposition_keywords(account.bio))
            pro_score += bio_pro * 2  # Bio keywords weighted higher
            anti_score += bio_anti * 2

        # Check tweets
        for tweet in tweets:
            if not tweet.text:
                continue
            tweet_pro = len(self.nlp.detect_pro_regime_keywords(tweet.text))
            tweet_anti = len(self.nlp.detect_opposition_keywords(tweet.text))
            pro_score += tweet_pro
            anti_score += tweet_anti

        # Determine stance based on scores
        total = pro_score + anti_score

        if total > 0:
            # Have keyword signals
            pro_ratio = pro_score / total
            if pro_ratio > 0.7:
                stance = PoliticalStance.PRO_REGIME
                taxonomy = self._determine_pro_taxonomy(account, tweets)
            elif pro_ratio < 0.3:
                stance = PoliticalStance.ANTI_REGIME
                taxonomy = self._determine_anti_taxonomy(account, tweets)
            else:
                stance = PoliticalStance.NEUTRAL
                taxonomy = None
            return stance, taxonomy

        # No keyword signals - use network-based classification
        # Only if account is an amplifier (found via following/retweeting regime content)
        if account.account_type == AccountType.AMPLIFIER:
            # Check if bio has any Persian/Farsi text (more likely to be engaged with Iran)
            has_persian = account.bio and any(
                '\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF'
                for c in (account.bio or "")
            )
            # Check if display name has Persian
            has_persian_name = account.display_name and any(
                '\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF'
                for c in (account.display_name or "")
            )

            if has_persian or has_persian_name:
                # Persian-speaking follower of regime = likely pro-regime
                return PoliticalStance.PRO_REGIME, PoliticalTaxonomy.UNKNOWN
            else:
                # Non-Persian follower = could be anyone, mark as neutral/unknown
                return PoliticalStance.NEUTRAL, None

        return PoliticalStance.UNKNOWN, None

    def _determine_seed_taxonomy(self, account: Account) -> PoliticalTaxonomy:
        """Determine taxonomy for seed accounts based on category."""
        if account.seed_category:
            cat = account.seed_category.value
            if cat == "irgc_linked":
                return PoliticalTaxonomy.IRGC_ALIGNED
            elif cat == "state_media":
                return PoliticalTaxonomy.IRGC_ALIGNED
            elif cat == "official_state":
                return PoliticalTaxonomy.PRINCIPLIST
        return PoliticalTaxonomy.UNKNOWN

    def _determine_pro_taxonomy(
        self, account: Account, tweets: list[Tweet]
    ) -> PoliticalTaxonomy:
        """Determine finer taxonomy for pro-regime accounts."""
        all_text = account.bio or ""
        for t in tweets:
            all_text += " " + (t.text or "")

        all_text_lower = all_text.lower()
        all_text_norm = self.nlp.normalize(all_text)

        # IRGC-aligned indicators
        irgc_keywords = ["سپاه", "بسیج", "حاج قاسم", "سلیمانی", "irgc", "basij"]
        if any(kw in all_text_lower or kw in all_text_norm for kw in irgc_keywords):
            return PoliticalTaxonomy.IRGC_ALIGNED

        # Principlist indicators
        principlist_keywords = ["اصولگرا", "principlist", "conservative"]
        if any(kw in all_text_lower or kw in all_text_norm for kw in principlist_keywords):
            return PoliticalTaxonomy.PRINCIPLIST

        # Reformist indicators
        reformist_keywords = ["اصلاح", "reformist", "reform"]
        if any(kw in all_text_lower or kw in all_text_norm for kw in reformist_keywords):
            return PoliticalTaxonomy.REFORMIST

        # Default to IRGC-aligned for state media
        if account.is_seed or account.account_type in [
            AccountType.OFFICIAL_STATE,
            AccountType.STATE_MEDIA,
            AccountType.IRGC_LINKED,
        ]:
            return PoliticalTaxonomy.IRGC_ALIGNED

        return PoliticalTaxonomy.UNKNOWN

    def _determine_anti_taxonomy(
        self, account: Account, tweets: list[Tweet]
    ) -> PoliticalTaxonomy:
        """Determine finer taxonomy for anti-regime accounts."""
        all_text = account.bio or ""
        for t in tweets:
            all_text += " " + (t.text or "")

        all_text_lower = all_text.lower()
        all_text_norm = self.nlp.normalize(all_text)

        # Monarchist indicators
        monarchist_keywords = ["شاه", "پهلوی", "رضا پهلوی", "shah", "pahlavi", "monarchist"]
        if any(kw in all_text_lower or kw in all_text_norm for kw in monarchist_keywords):
            return PoliticalTaxonomy.MONARCHIST

        # MEK indicators
        mek_keywords = ["مجاهدین", "رجوی", "mek", "pmoi", "ncri", "rajavi"]
        if any(kw in all_text_lower or kw in all_text_norm for kw in mek_keywords):
            return PoliticalTaxonomy.MEK

        # Leftist indicators
        leftist_keywords = ["کمونیست", "سوسیالیست", "چپ", "communist", "socialist", "leftist"]
        if any(kw in all_text_lower or kw in all_text_norm for kw in leftist_keywords):
            return PoliticalTaxonomy.LEFTIST

        # Republican/democratic indicators
        republican_keywords = ["جمهوری‌خواه", "دموکراسی", "republic", "democrat"]
        if any(kw in all_text_lower or kw in all_text_norm for kw in republican_keywords):
            return PoliticalTaxonomy.REPUBLICAN

        return PoliticalTaxonomy.UNKNOWN

    def classify_threat_level(
        self,
        account: Account,
        tweets: list[Tweet],
    ) -> ThreatLevel:
        """Classify threat level for pro-regime accounts.

        Args:
            account: Account to classify
            tweets: Account's tweets

        Returns:
            ThreatLevel classification
        """
        # Only classify pro-regime accounts
        if account.political_stance != PoliticalStance.PRO_REGIME:
            return ThreatLevel.UNKNOWN

        # Check if known IRGC media account
        if self.nlp.is_irgc_media_account(account.username):
            return ThreatLevel.STATE_PROPAGANDIST

        # Combine all text for analysis
        all_text = account.bio or ""
        for t in tweets:
            all_text += " " + (t.text or "")

        # Count threat signals across all tweets
        violence_count = 0
        irgc_count = 0
        doxxing_count = 0
        harassment_count = 0

        for t in tweets:
            if not t.text:
                continue
            signals = self.nlp.get_threat_signals(t.text)
            violence_count += len(signals["violence_keywords"])
            irgc_count += len(signals["irgc_signals"])
            doxxing_count += len(signals["doxxing_indicators"])
            harassment_count += len(signals["harassment_keywords"])

        # Also check bio
        bio_signals = self.nlp.get_threat_signals(account.bio or "")
        irgc_count += len(bio_signals["irgc_signals"])

        # Priority classification (most dangerous first)

        # 1. Violence inciter - immediate flag for calls to violence
        if violence_count >= 2:
            return ThreatLevel.VIOLENCE_INCITER

        # 2. Doxxer - exposes opposition identities
        if doxxing_count >= 2:
            return ThreatLevel.DOXXER

        # 3. IRGC operative - direct IRGC connection
        if irgc_count >= 3 or (irgc_count >= 1 and account.seed_category and "irgc" in account.seed_category.value):
            return ThreatLevel.IRGC_OPERATIVE

        # 4. State propagandist - official state media
        if account.is_seed and account.seed_category:
            if account.seed_category.value in ["official_state", "state_media"]:
                return ThreatLevel.STATE_PROPAGANDIST

        # 5. Check if likely a bot (amplifier bot)
        if account.bot_score and account.bot_score >= 0.7:
            return ThreatLevel.AMPLIFIER_BOT

        # 6. Troll - harassment campaigns
        if harassment_count >= 3:
            return ThreatLevel.TROLL

        # 7. Passive supporter - engages but doesn't incite
        if violence_count == 0 and harassment_count < 2:
            return ThreatLevel.PASSIVE_SUPPORTER

        return ThreatLevel.UNKNOWN

    def _count_account_threat_signals(
        self,
        account: Account,
        tweets: list[Tweet],
    ) -> dict:
        """Count all threat signals for an account.

        Returns dict with counts for each signal type.
        """
        counts = {
            "violence": 0,
            "irgc": 0,
            "doxxing": 0,
            "harassment": 0,
        }

        # Check bio
        bio_signals = self.nlp.get_threat_signals(account.bio or "")
        counts["violence"] += len(bio_signals["violence_keywords"])
        counts["irgc"] += len(bio_signals["irgc_signals"])
        counts["doxxing"] += len(bio_signals["doxxing_indicators"])
        counts["harassment"] += len(bio_signals["harassment_keywords"])

        # Check tweets
        for t in tweets:
            if not t.text:
                continue
            signals = self.nlp.get_threat_signals(t.text)
            counts["violence"] += len(signals["violence_keywords"])
            counts["irgc"] += len(signals["irgc_signals"])
            counts["doxxing"] += len(signals["doxxing_indicators"])
            counts["harassment"] += len(signals["harassment_keywords"])

        return counts

    def _llm_classify(
        self, account: Account, tweets: list[Tweet]
    ) -> tuple[PoliticalStance, Optional[PoliticalTaxonomy]]:
        """Classify using Claude LLM."""
        if not self.client:
            return PoliticalStance.UNKNOWN, None

        # Prepare context
        tweet_texts = [t.text for t in tweets[:20] if t.text]
        tweet_sample = "\n".join([f"- {t[:280]}" for t in tweet_texts])

        system_prompt = """You are an expert analyst on Iranian politics and social media.
Your task is to classify Twitter accounts based on their political stance regarding the Islamic Republic of Iran.

Political Stances:
- pro_regime: Supports the Islamic Republic, Supreme Leader, IRGC
- anti_regime: Opposes the Islamic Republic, supports protests/opposition
- neutral: No clear political stance or mixed positions

If pro_regime, also determine taxonomy:
- principlist: Traditional conservative supporters
- reformist: Supports reform within the system
- irgc_aligned: IRGC-affiliated or strongly supports IRGC

If anti_regime, also determine taxonomy:
- monarchist: Supports restoration of Pahlavi monarchy
- republican: Supports secular democratic republic
- mek: Affiliated with Mojahedin-e Khalq
- leftist: Socialist/communist opposition

Important context:
- Account may use sarcasm or coded language
- Persian text should be interpreted with cultural context
- State media accounts (PressTV, Tasnim, Fars, IRNA) are pro-regime by definition
- Look for regime talking points: "فتنه" (sedition), "اغتشاشگر" (rioters), "مقاومت" (resistance)
- Look for opposition signals: "زن زندگی آزادی" (Woman Life Freedom), protest support"""

        user_prompt = f"""Classify this Twitter account:

Username: @{account.username}
Display name: {account.display_name}
Bio: {account.bio or "(no bio)"}
Verified: {account.is_verified}

Sample tweets:
{tweet_sample}

Respond ONLY with a JSON object:
{{"stance": "pro_regime|anti_regime|neutral", "taxonomy": "principlist|reformist|irgc_aligned|monarchist|republican|mek|leftist|null", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        try:
            response = self.client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse response
            import json
            import re

            content = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                result = json.loads(json_match.group())
                stance_str = result.get("stance", "unknown")
                taxonomy_str = result.get("taxonomy")

                # Map to enums
                stance_map = {
                    "pro_regime": PoliticalStance.PRO_REGIME,
                    "anti_regime": PoliticalStance.ANTI_REGIME,
                    "neutral": PoliticalStance.NEUTRAL,
                }
                taxonomy_map = {
                    "principlist": PoliticalTaxonomy.PRINCIPLIST,
                    "reformist": PoliticalTaxonomy.REFORMIST,
                    "irgc_aligned": PoliticalTaxonomy.IRGC_ALIGNED,
                    "monarchist": PoliticalTaxonomy.MONARCHIST,
                    "republican": PoliticalTaxonomy.REPUBLICAN,
                    "mek": PoliticalTaxonomy.MEK,
                    "leftist": PoliticalTaxonomy.LEFTIST,
                }

                stance = stance_map.get(stance_str, PoliticalStance.UNKNOWN)
                taxonomy = taxonomy_map.get(taxonomy_str) if taxonomy_str else None

                return stance, taxonomy

        except Exception:
            pass

        return PoliticalStance.UNKNOWN, None

    def classify_all_accounts(
        self,
        account_ids: Optional[list[int]] = None,
        use_llm: bool = True,
    ) -> int:
        """Classify all accounts (or specified accounts).

        Returns number of accounts classified.
        """
        if account_ids is None:
            accounts = self.db.get_all_accounts()
        else:
            accounts = [self.db.get_account(aid) for aid in account_ids]
            accounts = [a for a in accounts if a]

        classified = 0
        for account in accounts:
            if not account.id:
                continue

            tweets = self.db.get_tweets_by_account(account.id, limit=50)
            stance, taxonomy, threat_level = self.classify_account(account, tweets, use_llm=use_llm)

            # Determine account type
            account_type = self._determine_account_type(account, stance)

            # Update database
            self.db.update_account_classification(
                account.id,
                account_type=account_type,
                political_stance=stance,
                political_taxonomy=taxonomy,
                threat_level=threat_level,
            )

            # Update regime score
            regime_score = self._stance_to_score(stance)
            self.db.update_account_scores(account.id, regime_score=regime_score)

            classified += 1

        return classified

    def get_threat_breakdown(self) -> dict[ThreatLevel, int]:
        """Get breakdown of pro-regime accounts by threat level.

        Returns dict mapping ThreatLevel to count.
        """
        accounts = self.db.get_all_accounts(stance=PoliticalStance.PRO_REGIME)
        breakdown = {level: 0 for level in ThreatLevel}

        for acc in accounts:
            breakdown[acc.threat_level] = breakdown.get(acc.threat_level, 0) + 1

        return breakdown

    def _determine_account_type(
        self, account: Account, stance: PoliticalStance
    ) -> AccountType:
        """Determine account type based on classification."""
        # Check if seed account (already classified)
        if account.is_seed and account.seed_category:
            type_map = {
                "official_state": AccountType.OFFICIAL_STATE,
                "state_media": AccountType.STATE_MEDIA,
                "irgc_linked": AccountType.IRGC_LINKED,
            }
            return type_map.get(account.seed_category.value, AccountType.UNKNOWN)

        # Check if likely bot
        if account.bot_score and account.bot_score > self.settings.bot_threshold:
            return AccountType.BOT

        # Pro-regime non-seeds are likely amplifiers
        if stance == PoliticalStance.PRO_REGIME:
            return AccountType.AMPLIFIER

        return AccountType.UNKNOWN

    def _stance_to_score(self, stance: PoliticalStance) -> float:
        """Convert stance to regime alignment score."""
        score_map = {
            PoliticalStance.PRO_REGIME: 1.0,
            PoliticalStance.ANTI_REGIME: 0.0,
            PoliticalStance.NEUTRAL: 0.5,
            PoliticalStance.UNKNOWN: 0.5,
        }
        return score_map.get(stance, 0.5)

    def get_accounts_by_stance(
        self, stance: PoliticalStance, limit: int = 100
    ) -> list[Account]:
        """Get accounts with a specific stance."""
        return self.db.get_all_accounts(stance=stance, limit=limit)
