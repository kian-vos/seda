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
        """
        clean_text = self.clean_for_analysis(text)

        return {
            "normalized_text": clean_text,
            "hashtags": self.extract_hashtags(text),
            "mentions": self.extract_mentions(text),
            "is_persian": self.is_persian(text),
            "persian_ratio": self.get_persian_ratio(text),
            "pro_regime_keywords": self.detect_pro_regime_keywords(text),
            "opposition_keywords": self.detect_opposition_keywords(text),
            "keyword_score": self.get_keyword_score(text),
        }


# Singleton instance
_nlp: Optional[PersianNLP] = None


def get_nlp() -> PersianNLP:
    """Get or create global PersianNLP instance."""
    global _nlp
    if _nlp is None:
        _nlp = PersianNLP()
    return _nlp
