"""Persian NLP preprocessing for SEDA."""

import re
from typing import Optional

# Try to import hazm, fallback to basic processing if not available
try:
    from hazm import Normalizer, word_tokenize, stopwords_list
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False


class PersianNLP:
    """Persian text preprocessing and analysis."""

    # Arabic to Persian character mappings
    ARABIC_TO_PERSIAN = {
        "ي": "ی",
        "ك": "ک",
        "٤": "۴",
        "٥": "۵",
        "٦": "۶",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "آ": "آ",
    }

    # Pro-regime keywords (Persian)
    PRO_REGIME_KEYWORDS_FA = [
        # Direct regime support
        "لبیک یا خامنه",  # "At your service, Khamenei"
        "خامنه ای",  # Khamenei
        "خامنه‌ای",  # Khamenei (with ZWNJ)
        "رهبر انقلاب",  # Leader of the Revolution
        "رهبر معظم",  # Supreme Leader
        "امام خامنه‌ای",  # Imam Khamenei
        "جمهوری اسلامی",  # Islamic Republic
        "جان بر کف",  # Ready to sacrifice life
        "لبیک",  # "At your service" (religious devotion)
        # Regime narratives
        "اغتشاشگر",  # Rioters
        "فتنه",  # Sedition
        "محور مقاومت",  # Axis of Resistance
        "استکبار",  # Arrogance (referring to West)
        "منافق",  # Hypocrites (regime term for MEK)
        "براندازان",  # Subversives
        "دشمن",  # Enemy
        # Religious/ideological
        "انقلاب اسلامی",  # Islamic Revolution
        "ولایت فقیه",  # Guardianship of the Jurist
        "شهید",  # Martyr
        "مقاومت",  # Resistance
        "نظام مقدس",  # Sacred system
        # Organizations
        "سپاه",  # IRGC
        "بسیج",  # Basij
        "حزب الله",  # Hezbollah
        # Anti-West/Israel
        "صهیونیست",  # Zionist
        "آمریکای جنایتکار",  # Criminal America
        "استکبار جهانی",  # Global arrogance
        # Shia religious (regime-aligned context)
        "یا حسین",  # O Hussein (when with regime signals)
        "یا مهدی",  # O Mahdi
        # Self-identification terms used by regime supporters
        "ارزشی",  # Value-oriented (regime supporter term)
        "انقلابی",  # Revolutionary (Islamic Revolution supporter)
        "اصولگرا",  # Principlist
        "حزب اللهی",  # Hezbollahi (regime loyalist)
        "خط امام",  # Imam's line (Khomeini's path)
    ]

    # Pro-regime keywords (English)
    PRO_REGIME_KEYWORDS_EN = [
        "islamic republic of iran",
        "islamic republic",
        "resistance axis",
        "axis of resistance",
        "zionist regime",
        "zionist entity",
        "arrogance",
        "martyr",
        "martyrdom",
        "islamic revolution",
        "imam khamenei",
        "supreme leader",
        "leader of the revolution",
        "irgc",
        "basij",
        "hezbollah",
        "hezballah",
        "american imperialism",
        "rioters",
        "sedition",
        "anti-zionist",
        "death to israel",
        "death to america",
    ]

    # Opposition keywords (Persian)
    OPPOSITION_KEYWORDS_FA = [
        "زن زندگی آزادی",  # Woman Life Freedom
        "مرگ بر دیکتاتور",  # Death to dictator
        "مرگ بر خامنه ای",  # Death to Khamenei
        "نه به جمهوری اسلامی",  # No to Islamic Republic
        "مهسا امینی",  # Mahsa Amini
        "ژینا",  # Jina (Mahsa's Kurdish name)
        "سرنگونی",  # Overthrow
        "دیکتاتور",  # Dictator
        "حکومت آخوندی",  # Mullah regime
        "جمهوری اسلامی نابود",  # Destroy Islamic Republic
        "مردم معترض",  # Protesting people
        "زندانی سیاسی",  # Political prisoner
        "براندازی",  # Regime change (opposition context)
        "آخوند",  # Mullah (pejorative)
        "ملایان",  # Mullahs (pejorative)
    ]

    # Opposition keywords (English)
    OPPOSITION_KEYWORDS_EN = [
        "woman life freedom",
        "mahsa amini",
        "jina amini",
        "death to dictator",
        "iran revolution",
        "free iran",
        "regime change",
        "political prisoner",
        "mullah regime",
        "islamic republic must go",
        "protesters",
        "uprising",
    ]

    # Violence inciter keywords (Persian)
    VIOLENCE_KEYWORDS_FA = [
        # Direct violence calls
        "اعدام کنید",        # Execute them
        "بکشید",            # Kill them
        "به درک",           # Send to hell
        "نابود کنید",       # Destroy them
        "تیرباران",         # Firing squad
        "دار بزنید",        # Hang them
        "اعدامشان",         # Execute them (variant)
        "بزنید",            # Beat them
        # Dehumanization
        "اوباش",            # Thugs
        "اراذل",            # Riffraff
        "حشرات",           # Insects
        "سگ",              # Dogs (pejorative)
        "موش",             # Rats
        "نجس",             # Impure/unclean
        # Incitement language
        "سرکوب کنید",       # Suppress them
        "جمع کنید",        # Round them up
        "پاکسازی",         # Cleanse/purge
        "ریشه کن",         # Eradicate
        # Targeting protesters
        "آشوبگر باید",      # Rioters must...
        "اغتشاشگر مرگ",     # Death to rioters
        "فتنه گر",         # Seditionist
        "مزدور باید",       # Mercenaries must...
        "ضد انقلاب",        # Counter-revolutionary
    ]

    # Violence inciter keywords (English)
    VIOLENCE_KEYWORDS_EN = [
        "execute them",
        "hang them",
        "kill them",
        "death to rioters",
        "firing squad",
        "eliminate",
        "destroy protesters",
        "crush the uprising",
        "no mercy",
        "punish rioters",
        "make example of",
        "traitors deserve",
    ]

    # IRGC strong signals (Persian)
    IRGC_STRONG_SIGNALS_FA = [
        # Organizations
        "سپاه پاسداران",     # IRGC full name
        "نیروی قدس",        # Quds Force
        "بسیج مستضعفین",   # Basij full name
        "حشد الشعبی",      # Iraqi PMF (IRGC proxy)
        # Commanders and figures
        "سلیمانی",         # Soleimani
        "حاج قاسم",        # Haj Qasem
        "قاآنی",           # Qaani (current Quds commander)
        "جعفری",          # Jafari (former IRGC commander)
        "شهید سلیمانی",    # Martyr Soleimani
        "سردار سلیمانی",   # Commander Soleimani
        # Proxies
        "حزب الله",        # Hezbollah
        "انصارالله",       # Houthis
        "حماس",           # Hamas
        "جهاد اسلامی",     # Islamic Jihad
        "فاطمیون",        # Fatemiyoun (Afghan fighters)
        "زینبیون",        # Zainabiyoun (Pakistani fighters)
        # IRGC terminology
        "مدافعان حرم",     # Defenders of the shrine
        "محور مقاومت",     # Axis of Resistance
        "جبهه مقاومت",     # Resistance front
    ]

    # IRGC strong signals (English)
    IRGC_STRONG_SIGNALS_EN = [
        "irgc",
        "islamic revolutionary guard",
        "quds force",
        "basij",
        "qasem soleimani",
        "esmail qaani",
        "axis of resistance",
        "resistance axis",
        "hezbollah",
        "houthis",
        "ansarallah",
        "hamas",
        "islamic jihad",
        "fatemiyoun",
        "zainabiyoun",
        "pmf",
        "popular mobilization",
    ]

    # Known IRGC-affiliated media accounts
    IRGC_MEDIA_ACCOUNTS = [
        "farsnews_agency",
        "tasnimnews_en",
        "sepahnews",
        "press_tv",
        "presstv",
        "iraborhan",
        "khaboronline",
        "isaborhan",
        "iikirgc",
    ]

    # Doxxing indicators (Persian) - more specific phrases to avoid false positives
    DOXXING_KEYWORDS_FA = [
        "شناسایی شد",      # Has been identified
        "هویت واقعی",      # Real identity
        "آدرس منزل",       # Home address
        "محل کار",        # Workplace
        "افشا شد",        # Has been exposed
        "معرفی می‌کنیم",   # We introduce (expose)
        "مشخصات کامل",    # Full details
        "خانواده‌اش",      # Their family (possessive - more likely doxxing context)
        "نام واقعی",      # Real name
        "شناسایی کنید",   # Identify them (call to action)
    ]

    # Doxxing indicators (English)
    DOXXING_KEYWORDS_EN = [
        "identified as",
        "real name is",
        "lives at",
        "works at",
        "their family",    # More specific than just "family members"
        "exposed",
        "doxxed",
        "personal information",
        "home address",
    ]

    # Harassment/troll indicators (Persian)
    HARASSMENT_KEYWORDS_FA = [
        "خفه شو",          # Shut up
        "گمشو",            # Get lost
        "نمک‌گیر",         # Sellout
        "وطن‌فروش",        # Traitor
        "خائن",            # Traitor
        "بی‌ناموس",        # Dishonorable
        "بی‌شرف",          # Shameless
        "مزدور",           # Mercenary
    ]

    # Harassment/troll indicators (English)
    HARASSMENT_KEYWORDS_EN = [
        "shut up",
        "go to hell",
        "traitor",
        "sellout",
        "paid agent",
        "foreign agent",
        "cia agent",
        "mossad agent",
    ]

    def __init__(self):
        """Initialize Persian NLP processor."""
        self.normalizer = Normalizer() if HAZM_AVAILABLE else None
        self.stopwords = set(stopwords_list()) if HAZM_AVAILABLE else set()

        # Compile regex patterns
        self._arabic_pattern = re.compile(
            "[" + "".join(self.ARABIC_TO_PERSIAN.keys()) + "]"
        )
        self._hashtag_pattern = re.compile(r"#(\w+)", re.UNICODE)
        self._mention_pattern = re.compile(r"@(\w+)", re.UNICODE)
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+", re.UNICODE
        )
        self._persian_pattern = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

    def normalize(self, text: str) -> str:
        """Normalize Persian text.

        - Convert Arabic characters to Persian
        - Handle ZWNJ (نیم‌فاصله)
        - Normalize spacing
        """
        if not text:
            return ""

        # Convert Arabic to Persian
        text = self._arabic_pattern.sub(
            lambda m: self.ARABIC_TO_PERSIAN.get(m.group(), m.group()),
            text,
        )

        # Use hazm normalizer if available
        if self.normalizer:
            text = self.normalizer.normalize(text)
        else:
            # Basic normalization
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize Persian text."""
        if not text:
            return []

        normalized = self.normalize(text)

        if HAZM_AVAILABLE:
            return word_tokenize(normalized)
        else:
            # Basic tokenization
            return normalized.split()

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove Persian stopwords from token list."""
        return [t for t in tokens if t not in self.stopwords]

    def extract_hashtags(self, text: str) -> list[str]:
        """Extract hashtags from text."""
        if not text:
            return []
        return self._hashtag_pattern.findall(text)

    def extract_mentions(self, text: str) -> list[str]:
        """Extract @mentions from text."""
        if not text:
            return []
        return self._mention_pattern.findall(text)

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        if not text:
            return ""
        return self._url_pattern.sub("", text)

    def is_persian(self, text: str) -> bool:
        """Check if text contains Persian characters."""
        if not text:
            return False
        return bool(self._persian_pattern.search(text))

    def get_persian_ratio(self, text: str) -> float:
        """Get ratio of Persian characters in text."""
        if not text:
            return 0.0

        persian_chars = len(self._persian_pattern.findall(text))
        total_chars = len(re.sub(r"\s", "", text))

        if total_chars == 0:
            return 0.0

        return persian_chars / total_chars

    def detect_pro_regime_keywords(self, text: str) -> list[str]:
        """Detect pro-regime keywords in text."""
        if not text:
            return []

        text_lower = text.lower()
        text_normalized = self.normalize(text)

        found = []

        # Check Persian keywords
        for keyword in self.PRO_REGIME_KEYWORDS_FA:
            if keyword in text_normalized:
                found.append(keyword)

        # Check English keywords
        for keyword in self.PRO_REGIME_KEYWORDS_EN:
            if keyword in text_lower:
                found.append(keyword)

        return found

    def detect_opposition_keywords(self, text: str) -> list[str]:
        """Detect opposition keywords in text."""
        if not text:
            return []

        text_lower = text.lower()
        text_normalized = self.normalize(text)

        found = []

        # Check Persian keywords
        for keyword in self.OPPOSITION_KEYWORDS_FA:
            if keyword in text_normalized:
                found.append(keyword)

        # Check English keywords
        for keyword in self.OPPOSITION_KEYWORDS_EN:
            if keyword in text_lower:
                found.append(keyword)

        return found

    def get_keyword_score(self, text: str) -> float:
        """Get keyword-based regime alignment score.

        Returns:
            Score from -1 (anti-regime) to 1 (pro-regime), 0 is neutral
        """
        if not text:
            return 0.0

        pro_keywords = self.detect_pro_regime_keywords(text)
        opp_keywords = self.detect_opposition_keywords(text)

        pro_count = len(pro_keywords)
        opp_count = len(opp_keywords)

        total = pro_count + opp_count
        if total == 0:
            return 0.0

        return (pro_count - opp_count) / total

    def clean_for_analysis(self, text: str) -> str:
        """Clean text for analysis - remove URLs, normalize."""
        if not text:
            return ""

        text = self.remove_urls(text)
        text = self.normalize(text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def detect_violence_keywords(self, text: str) -> list[str]:
        """Detect violence inciting keywords in text.

        Returns list of matched violence keywords.
        """
        if not text:
            return []

        text_lower = text.lower()
        text_normalized = self.normalize(text)

        found = []

        # Check Persian keywords
        for keyword in self.VIOLENCE_KEYWORDS_FA:
            if keyword in text_normalized:
                found.append(keyword)

        # Check English keywords
        for keyword in self.VIOLENCE_KEYWORDS_EN:
            if keyword in text_lower:
                found.append(keyword)

        return found

    def detect_irgc_signals(self, text: str) -> list[str]:
        """Detect IRGC-related signals in text.

        Returns list of matched IRGC keywords/signals.
        """
        if not text:
            return []

        text_lower = text.lower()
        text_normalized = self.normalize(text)

        found = []

        # Check Persian keywords
        for keyword in self.IRGC_STRONG_SIGNALS_FA:
            if keyword in text_normalized:
                found.append(keyword)

        # Check English keywords
        for keyword in self.IRGC_STRONG_SIGNALS_EN:
            if keyword in text_lower:
                found.append(keyword)

        return found

    def detect_doxxing_indicators(self, text: str) -> list[str]:
        """Detect doxxing indicators in text.

        Returns list of matched doxxing keywords.
        """
        if not text:
            return []

        text_lower = text.lower()
        text_normalized = self.normalize(text)

        found = []

        # Check Persian keywords
        for keyword in self.DOXXING_KEYWORDS_FA:
            if keyword in text_normalized:
                found.append(keyword)

        # Check English keywords
        for keyword in self.DOXXING_KEYWORDS_EN:
            if keyword in text_lower:
                found.append(keyword)

        return found

    def detect_harassment_keywords(self, text: str) -> list[str]:
        """Detect harassment/troll keywords in text.

        Returns list of matched harassment keywords.
        """
        if not text:
            return []

        text_lower = text.lower()
        text_normalized = self.normalize(text)

        found = []

        # Check Persian keywords
        for keyword in self.HARASSMENT_KEYWORDS_FA:
            if keyword in text_normalized:
                found.append(keyword)

        # Check English keywords
        for keyword in self.HARASSMENT_KEYWORDS_EN:
            if keyword in text_lower:
                found.append(keyword)

        return found

    def is_irgc_media_account(self, username: str) -> bool:
        """Check if username is a known IRGC-affiliated media account."""
        if not username:
            return False
        return username.lower() in self.IRGC_MEDIA_ACCOUNTS

    def get_violence_score(self, text: str) -> float:
        """Get violence incitement score.

        Returns:
            Score from 0 (no violence) to 1 (high violence incitement)
        """
        if not text:
            return 0.0

        violence_keywords = self.detect_violence_keywords(text)
        count = len(violence_keywords)

        # Threshold: >= 2 keywords is high concern
        if count >= 3:
            return 1.0
        elif count == 2:
            return 0.8
        elif count == 1:
            return 0.4
        return 0.0

    def get_threat_signals(self, text: str) -> dict:
        """Get all threat-related signals from text.

        Returns dict with:
            - violence_keywords: list of violence keywords found
            - irgc_signals: list of IRGC signals found
            - doxxing_indicators: list of doxxing indicators found
            - harassment_keywords: list of harassment keywords found
            - violence_score: violence incitement score (0-1)
        """
        return {
            "violence_keywords": self.detect_violence_keywords(text),
            "irgc_signals": self.detect_irgc_signals(text),
            "doxxing_indicators": self.detect_doxxing_indicators(text),
            "harassment_keywords": self.detect_harassment_keywords(text),
            "violence_score": self.get_violence_score(text),
        }

    def preprocess_tweet(self, text: str) -> dict:
        """Preprocess a tweet and extract features.

        Returns dict with:
            - normalized_text: cleaned text
            - hashtags: list of hashtags
            - mentions: list of mentions
            - is_persian: whether text is Persian
            - persian_ratio: ratio of Persian characters
            - pro_regime_keywords: list of detected pro-regime keywords
            - opposition_keywords: list of detected opposition keywords
            - keyword_score: alignment score
            - violence_keywords: list of violence incitement keywords
            - irgc_signals: list of IRGC-related signals
            - doxxing_indicators: list of doxxing indicators
            - harassment_keywords: list of harassment keywords
            - violence_score: violence incitement score (0-1)
        """
        clean_text = self.clean_for_analysis(text)
        threat_signals = self.get_threat_signals(text)

        return {
            "normalized_text": clean_text,
            "hashtags": self.extract_hashtags(text),
            "mentions": self.extract_mentions(text),
            "is_persian": self.is_persian(text),
            "persian_ratio": self.get_persian_ratio(text),
            "pro_regime_keywords": self.detect_pro_regime_keywords(text),
            "opposition_keywords": self.detect_opposition_keywords(text),
            "keyword_score": self.get_keyword_score(text),
            **threat_signals,
        }


# Singleton instance
_nlp: Optional[PersianNLP] = None


def get_nlp() -> PersianNLP:
    """Get or create global PersianNLP instance."""
    global _nlp
    if _nlp is None:
        _nlp = PersianNLP()
    return _nlp
