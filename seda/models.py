"""Pydantic data models for SEDA."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AccountType(str, Enum):
    """Type of Twitter account."""

    OFFICIAL_STATE = "official_state"
    STATE_MEDIA = "state_media"
    IRGC_LINKED = "irgc_linked"
    AMPLIFIER = "amplifier"
    BOT = "bot"
    UNKNOWN = "unknown"


class PoliticalStance(str, Enum):
    """Political stance classification."""

    PRO_REGIME = "pro_regime"
    ANTI_REGIME = "anti_regime"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class PoliticalTaxonomy(str, Enum):
    """Finer political taxonomy."""

    # Pro-regime
    PRINCIPLIST = "principlist"
    REFORMIST = "reformist"
    IRGC_ALIGNED = "irgc_aligned"

    # Opposition
    MONARCHIST = "monarchist"
    REPUBLICAN = "republican"
    MEK = "mek"
    LEFTIST = "leftist"

    UNKNOWN = "unknown"


class SeedCategory(str, Enum):
    """Category for seed accounts."""

    OFFICIAL_STATE = "official_state"
    STATE_MEDIA = "state_media"
    IRGC_LINKED = "irgc_linked"
    OTHER = "other"


class CoordinationType(str, Enum):
    """Type of coordination detected."""

    SYNCHRONIZED_POSTING = "synchronized_posting"
    HASHTAG_CAMPAIGN = "hashtag_campaign"
    AMPLIFICATION_NETWORK = "amplification_network"


class Account(BaseModel):
    """Twitter account model."""

    id: Optional[int] = None
    twitter_id: str
    username: str
    display_name: str = ""
    bio: str = ""
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    created_at: Optional[datetime] = None
    profile_image_url: str = ""
    is_verified: bool = False
    location: str = ""

    # Analysis scores
    bot_score: Optional[float] = None
    regime_score: Optional[float] = None
    coordination_score: Optional[float] = None

    # Classifications
    account_type: AccountType = AccountType.UNKNOWN
    political_stance: PoliticalStance = PoliticalStance.UNKNOWN
    political_taxonomy: PoliticalTaxonomy = PoliticalTaxonomy.UNKNOWN

    # Seed tracking
    is_seed: bool = False
    seed_category: Optional[SeedCategory] = None

    # Timestamps
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Features (stored as JSON)
    features: Optional[dict] = None


class Tweet(BaseModel):
    """Tweet model."""

    id: Optional[int] = None
    tweet_id: str
    account_id: int
    text: str = ""
    created_at: Optional[datetime] = None

    # Content
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)

    # Engagement
    retweet_count: int = 0
    like_count: int = 0
    reply_count: int = 0

    # Tweet type
    is_retweet: bool = False
    is_reply: bool = False
    is_quote: bool = False
    referenced_tweet_id: Optional[str] = None

    # Analysis
    sentiment: Optional[float] = None
    regime_alignment: Optional[float] = None
    talking_points: list[str] = Field(default_factory=list)

    collected_at: Optional[datetime] = None


class CoordinationCluster(BaseModel):
    """Detected coordination cluster."""

    id: Optional[int] = None
    cluster_type: CoordinationType
    created_at: Optional[datetime] = None
    member_account_ids: list[int] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
    confidence_score: float = 0.0
    description: str = ""
    is_active: bool = True


class SeedAccount(BaseModel):
    """Seed account for initial data collection."""

    id: Optional[int] = None
    username: str
    category: SeedCategory
    added_at: Optional[datetime] = None
    notes: str = ""


class ScraperResult(BaseModel):
    """Result from Apify scraper."""

    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    items_count: int = 0


class AnalysisResult(BaseModel):
    """Result from analysis pipeline."""

    account_id: int
    bot_score: Optional[float] = None
    stance: Optional[PoliticalStance] = None
    taxonomy: Optional[PoliticalTaxonomy] = None
    coordination_clusters: list[int] = Field(default_factory=list)
    features: Optional[dict] = None
