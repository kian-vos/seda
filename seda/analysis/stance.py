"""Stance classification for SEDA."""

import time
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


def retry_on_connection_error(func, max_retries=3, base_delay=1.0):
    """Retry a function on connection errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            is_connection_error = any(x in error_str for x in [
                "connection reset", "connection refused", "timed out",
                "timeout", "http error", "network"
            ])
            if is_connection_error and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise

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

        # Classify threat level for pro-regime accounts (hybrid: keywords + LLM for ambiguous)
        threat_level = self.classify_threat_level(account_with_stance, tweets or [], use_llm=use_llm)

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
        use_llm: bool = False,
    ) -> ThreatLevel:
        """Classify threat level for pro-regime accounts.

        Uses hybrid approach: keywords first, LLM for ambiguous cases.

        Args:
            account: Account to classify
            tweets: Account's tweets
            use_llm: Whether to use LLM for ambiguous cases

        Returns:
            ThreatLevel classification
        """
        # Only classify pro-regime accounts
        if account.political_stance != PoliticalStance.PRO_REGIME:
            return ThreatLevel.UNKNOWN

        # PRIORITY 1: Seed accounts with known categories take precedence
        # State media/official accounts are propagandists, not operatives
        # (They report on IRGC but aren't IRGC themselves)
        if account.is_seed and account.seed_category:
            if account.seed_category.value in ["official_state", "state_media"]:
                return ThreatLevel.STATE_PROPAGANDIST
            if account.seed_category.value == "irgc_linked":
                return ThreatLevel.IRGC_OPERATIVE

        # Check if known IRGC media account (from our list)
        if self.nlp.is_irgc_media_account(account.username):
            return ThreatLevel.STATE_PROPAGANDIST

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

        signal_counts = {
            "violence": violence_count,
            "irgc": irgc_count,
            "doxxing": doxxing_count,
            "harassment": harassment_count,
        }

        # Priority classification for non-seed accounts (most dangerous first)

        # 2. Violence inciter - immediate flag for calls to violence
        if violence_count >= 2:
            return ThreatLevel.VIOLENCE_INCITER

        # 3. Doxxer - exposes opposition identities
        if doxxing_count >= 2:
            return ThreatLevel.DOXXER

        # 4. IRGC operative - direct IRGC connection (non-seed only)
        # Requires strong signal: many IRGC references or explicit IRGC affiliation in bio
        if irgc_count >= 5:
            return ThreatLevel.IRGC_OPERATIVE

        # 5. Check if likely a bot (amplifier bot)
        if account.bot_score and account.bot_score >= 0.7:
            return ThreatLevel.AMPLIFIER_BOT

        # 6. Troll - harassment campaigns
        if harassment_count >= 3:
            return ThreatLevel.TROLL

        # HYBRID: Use LLM for all accounts with tweets (not just ambiguous)
        # LLM can detect context, sarcasm, and coded language that keywords miss
        if use_llm and self.client and tweets:
            llm_threat = self._llm_classify_threat(account, tweets, signal_counts)
            if llm_threat != ThreatLevel.UNKNOWN:
                return llm_threat

        # Fallback to keyword-based classification
        # 7. Passive supporter - engages but doesn't incite
        if violence_count == 0 and harassment_count < 2:
            return ThreatLevel.PASSIVE_SUPPORTER

        return ThreatLevel.UNKNOWN

    def _llm_classify_threat(
        self,
        account: Account,
        tweets: list[Tweet],
        signal_counts: dict,
    ) -> ThreatLevel:
        """Use LLM to classify threat level for ambiguous cases.

        Args:
            account: Account to classify
            tweets: Account's tweets
            signal_counts: Dict of keyword signal counts

        Returns:
            ThreatLevel from LLM or UNKNOWN if failed
        """
        if not self.client:
            return ThreatLevel.UNKNOWN

        # Prepare tweet samples
        tweet_samples = []
        for t in tweets[:15]:
            if t.text:
                tweet_samples.append(f"- {t.text[:250]}")

        prompt = f"""Analyze this Iranian Twitter account and classify its threat level.

Account: @{account.username}
Display Name: {account.display_name or 'N/A'}
Bio: {account.bio or 'N/A'}
Followers: {account.followers_count:,}

Keyword signals detected:
- Violence keywords: {signal_counts['violence']}
- IRGC/military references: {signal_counts['irgc']}
- Doxxing indicators: {signal_counts['doxxing']}
- Harassment keywords: {signal_counts['harassment']}

Sample tweets:
{chr(10).join(tweet_samples)}

Threat levels (choose ONE):
- violence_inciter: Explicitly calls for violence, death, execution of protesters/opposition
- doxxer: Exposes real identities, addresses, workplaces of opposition figures
- irgc_operative: Shows direct IRGC/Basij military affiliation (not just mentions)
- state_propagandist: Official government messaging, state media narrative
- troll: Systematic harassment, insults, coordinated attacks on individuals
- passive_supporter: Supports regime but no incitement or harassment

Consider context and intent. Persian sarcasm mocking protesters is pro-regime.
Reporting on IRGC news is NOT being an operative.

Return ONLY JSON: {{"threat_level": "...", "confidence": 0.0-1.0, "reasoning": "brief"}}"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Fast and cheap
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            import json
            import re

            content = response.content[0].text
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                result = json.loads(json_match.group())
                threat_str = result.get("threat_level", "").lower()
                confidence = result.get("confidence", 0)

                # Only accept high-confidence results
                if confidence >= 0.6:
                    threat_map = {
                        "violence_inciter": ThreatLevel.VIOLENCE_INCITER,
                        "doxxer": ThreatLevel.DOXXER,
                        "irgc_operative": ThreatLevel.IRGC_OPERATIVE,
                        "state_propagandist": ThreatLevel.STATE_PROPAGANDIST,
                        "amplifier_bot": ThreatLevel.AMPLIFIER_BOT,
                        "troll": ThreatLevel.TROLL,
                        "passive_supporter": ThreatLevel.PASSIVE_SUPPORTER,
                    }
                    return threat_map.get(threat_str, ThreatLevel.UNKNOWN)

        except Exception:
            pass

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
        batch_size: int = 100,
        skip_inactive_days: int = 180,
        chunk_size: int = 500,
    ) -> int:
        """Classify all accounts (or specified accounts).

        Args:
            account_ids: Specific accounts to classify (all if None)
            use_llm: Whether to use LLM for ambiguous cases
            batch_size: Number of updates per batch commit
            skip_inactive_days: Skip accounts with no tweets in this many days (0=don't skip)
            chunk_size: Number of accounts to load from DB at a time

        Returns number of accounts classified.
        """
        from datetime import datetime, timezone, timedelta

        classified = 0
        skipped_inactive = 0
        batch_updates = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=skip_inactive_days) if skip_inactive_days > 0 else None

        def process_account(account: Account) -> bool:
            """Process a single account. Returns True if classified, False if skipped."""
            nonlocal classified, skipped_inactive, batch_updates

            if not account.id:
                return False

            # Use retry logic for database calls
            try:
                tweets = retry_on_connection_error(
                    lambda aid=account.id: self.db.get_tweets_by_account(aid, limit=50)
                )
            except Exception as e:
                print(f"  Warning: Failed to get tweets for {account.username}: {e}")
                tweets = []

            # Skip inactive accounts (no recent tweets)
            if cutoff_date and tweets:
                most_recent = max((t.created_at for t in tweets if t.created_at), default=None)
                if most_recent and most_recent < cutoff_date:
                    skipped_inactive += 1
                    return False
            elif cutoff_date and not tweets:
                # No tweets at all - skip
                skipped_inactive += 1
                return False

            stance, taxonomy, threat_level = self.classify_account(account, tweets, use_llm=use_llm)

            # Determine account type
            account_type = self._determine_account_type(account, stance)

            # Calculate regime score
            regime_score = self._stance_to_score(stance)

            # Add to batch
            batch_updates.append({
                "account_id": account.id,
                "account_type": account_type,
                "political_stance": stance,
                "political_taxonomy": taxonomy,
                "threat_level": threat_level,
                "regime_score": regime_score,
            })

            classified += 1

            # Commit batch when it reaches batch_size
            if len(batch_updates) >= batch_size:
                try:
                    retry_on_connection_error(
                        lambda updates=list(batch_updates): self.db.update_accounts_classification_batch(updates)
                    )
                except Exception as e:
                    print(f"  Warning: Failed to save batch: {e}")
                print(f"  Processed {classified} accounts (skipped {skipped_inactive} inactive)...")
                batch_updates = []

            return True

        # Process accounts - either specific IDs or all accounts in chunks
        if account_ids is not None:
            # Process specific accounts
            for aid in account_ids:
                try:
                    account = retry_on_connection_error(lambda a=aid: self.db.get_account(a))
                    if account:
                        process_account(account)
                except Exception as e:
                    print(f"  Warning: Failed to load account {aid}: {e}")
        else:
            # Optimization: Only load accounts that have tweets (huge performance gain)
            # First, get list of account IDs that have tweets
            try:
                with self.db.connection() as conn:
                    account_ids_with_tweets = [
                        row[0] for row in conn.execute(
                            "SELECT DISTINCT account_id FROM tweets"
                        ).fetchall()
                    ]
                print(f"  Found {len(account_ids_with_tweets)} accounts with tweets")
            except Exception as e:
                print(f"  Warning: Could not get accounts with tweets: {e}")
                account_ids_with_tweets = None

            if account_ids_with_tweets is not None:
                # Process only accounts with tweets
                for aid in account_ids_with_tweets:
                    try:
                        account = retry_on_connection_error(lambda a=aid: self.db.get_account(a))
                        if account:
                            process_account(account)
                    except Exception as e:
                        print(f"  Warning: Failed to load account {aid}: {e}")
            else:
                # Fallback to chunked loading for all accounts
                offset = 0
                while True:
                    try:
                        accounts = retry_on_connection_error(
                            lambda o=offset: self.db.get_all_accounts(limit=chunk_size, offset=o)
                        )
                    except Exception as e:
                        print(f"  Error loading accounts at offset {offset}: {e}")
                        break

                    if not accounts:
                        break

                    print(f"  Loaded {len(accounts)} accounts (offset {offset})...")
                    for account in accounts:
                        process_account(account)

                    offset += chunk_size

        # Commit remaining updates
        if batch_updates:
            try:
                retry_on_connection_error(
                    lambda updates=list(batch_updates): self.db.update_accounts_classification_batch(updates)
                )
            except Exception as e:
                print(f"  Warning: Failed to save final batch: {e}")

        if skipped_inactive > 0:
            print(f"  Skipped {skipped_inactive} inactive accounts (no activity in {skip_inactive_days} days)")

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

    def explain_account(
        self,
        account: Account,
        tweets: Optional[list[Tweet]] = None,
    ) -> dict:
        """Generate a detailed explanation of why an account was classified.

        Uses LLM to provide human-readable explanation with examples.

        Args:
            account: Account to explain
            tweets: Account's tweets (fetched if not provided)

        Returns:
            Dict with explanation, evidence, and example tweets
        """
        if tweets is None and account.id:
            tweets = self.db.get_tweets_by_account(account.id, limit=50)

        tweets = tweets or []

        # Get threat signals
        signal_counts = self._count_account_threat_signals(account, tweets)

        # Find example tweets with signals
        violence_examples = []
        irgc_examples = []
        doxxing_examples = []
        harassment_examples = []

        for t in tweets:
            if not t.text:
                continue
            signals = self.nlp.get_threat_signals(t.text)
            if signals["violence_keywords"] and len(violence_examples) < 3:
                violence_examples.append({
                    "text": t.text[:300],
                    "keywords": signals["violence_keywords"],
                    "date": t.created_at.strftime("%Y-%m-%d") if t.created_at else "Unknown"
                })
            if signals["irgc_signals"] and len(irgc_examples) < 3:
                irgc_examples.append({
                    "text": t.text[:300],
                    "keywords": signals["irgc_signals"],
                    "date": t.created_at.strftime("%Y-%m-%d") if t.created_at else "Unknown"
                })
            if signals["doxxing_indicators"] and len(doxxing_examples) < 3:
                doxxing_examples.append({
                    "text": t.text[:300],
                    "keywords": signals["doxxing_indicators"],
                    "date": t.created_at.strftime("%Y-%m-%d") if t.created_at else "Unknown"
                })
            if signals["harassment_keywords"] and len(harassment_examples) < 3:
                harassment_examples.append({
                    "text": t.text[:300],
                    "keywords": signals["harassment_keywords"],
                    "date": t.created_at.strftime("%Y-%m-%d") if t.created_at else "Unknown"
                })

        # Build rule-based explanation
        explanation_parts = []
        threat_level = account.threat_level.value if account.threat_level else "unknown"

        if threat_level == "violence_inciter":
            explanation_parts.append(f"This account has been flagged for inciting violence with {signal_counts['violence']} detected instances of violent language.")
        elif threat_level == "doxxer":
            explanation_parts.append(f"This account shows {signal_counts['doxxing']} indicators of doxxing behavior (exposing identities).")
        elif threat_level == "irgc_operative":
            explanation_parts.append(f"This account shows {signal_counts['irgc']} connections to IRGC/military narratives.")
        elif threat_level == "state_propagandist":
            explanation_parts.append("This account is associated with official state or state media based on seed categorization.")
        elif threat_level == "amplifier_bot":
            explanation_parts.append(f"This account has a high bot score ({account.bot_score:.2f}) indicating automated behavior.")
        elif threat_level == "troll":
            explanation_parts.append(f"This account shows {signal_counts['harassment']} instances of harassment language.")
        elif threat_level == "passive_supporter":
            explanation_parts.append("This account engages with pro-regime content but doesn't show violent or harassment patterns.")

        # Calculate activity score
        recent_tweets = [t for t in tweets if t.created_at]
        if recent_tweets:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            days_since_last = (now - max(t.created_at for t in recent_tweets)).days
            activity_status = "Active" if days_since_last < 30 else f"Inactive ({days_since_last} days since last tweet)"
        else:
            activity_status = "Unknown activity"

        result = {
            "username": account.username,
            "threat_level": threat_level,
            "political_stance": account.political_stance.value if account.political_stance else "unknown",
            "activity_status": activity_status,
            "signal_counts": signal_counts,
            "rule_based_explanation": " ".join(explanation_parts),
            "examples": {
                "violence": violence_examples,
                "irgc": irgc_examples,
                "doxxing": doxxing_examples,
                "harassment": harassment_examples,
            },
        }

        # Generate LLM explanation if available
        if self.client and tweets:
            llm_explanation = self._generate_llm_explanation(account, tweets, signal_counts, threat_level)
            result["llm_explanation"] = llm_explanation

        return result

    def _generate_llm_explanation(
        self,
        account: Account,
        tweets: list[Tweet],
        signal_counts: dict,
        threat_level: str,
    ) -> str:
        """Generate LLM-powered explanation for the classification."""
        if not self.client:
            return ""

        # Prepare tweet samples
        tweet_samples = []
        for t in tweets[:10]:
            if t.text:
                tweet_samples.append(f"- [{t.created_at.strftime('%Y-%m-%d') if t.created_at else 'Unknown'}]: {t.text[:200]}")

        prompt = f"""Analyze this Twitter/X account and explain why it has been classified as a "{threat_level}" in our Iranian regime propaganda detection system.

Account: @{account.username}
Display Name: {account.display_name or 'N/A'}
Bio: {account.bio or 'N/A'}
Followers: {account.followers_count:,}
Classification: {threat_level}

Detected signals:
- Violence keywords: {signal_counts['violence']}
- IRGC/military references: {signal_counts['irgc']}
- Doxxing indicators: {signal_counts['doxxing']}
- Harassment keywords: {signal_counts['harassment']}

Sample tweets:
{chr(10).join(tweet_samples)}

Provide a 2-3 sentence explanation of why this account was classified this way, citing specific evidence from the bio or tweets. Be factual and objective. If the classification seems questionable based on the evidence, note that."""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating explanation: {e}"
