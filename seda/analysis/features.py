"""Feature extraction for bot detection and analysis."""

import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from seda.db import get_db
from seda.models import Account, Tweet


class FeatureExtractor:
    """Extract features from accounts and tweets for bot detection."""

    # Iran timezone offset (UTC+3:30)
    IRAN_TZ_OFFSET = timedelta(hours=3, minutes=30)

    def __init__(self):
        self.db = get_db()

    def extract_account_features(self, account: Account, tweets: list[Tweet]) -> dict:
        """Extract all features for an account.

        Returns a dictionary with ~40 features for bot detection.
        """
        features = {}

        # Profile features
        features.update(self._extract_profile_features(account))

        # Timing features (require tweets)
        if tweets:
            features.update(self._extract_timing_features(tweets))
            features.update(self._extract_content_features(tweets))
            features.update(self._extract_engagement_features(tweets))
        else:
            # Default values when no tweets
            features.update(self._default_timing_features())
            features.update(self._default_content_features())
            features.update(self._default_engagement_features())

        return features

    def _extract_profile_features(self, account: Account) -> dict:
        """Extract profile-based features."""
        features = {}

        username = account.username or ""

        # Username digit ratio
        digits = sum(c.isdigit() for c in username)
        features["username_digit_ratio"] = digits / len(username) if username else 0

        # Username length
        features["username_length"] = len(username)

        # Bio length (0 can be suspicious)
        features["bio_length"] = len(account.bio or "")
        features["has_bio"] = 1 if account.bio else 0

        # Default avatar check (heuristic: empty or default URL patterns)
        profile_img = account.profile_image_url or ""
        features["has_default_avatar"] = 1 if "default" in profile_img.lower() or not profile_img else 0

        # Account age
        if account.created_at:
            age_days = (datetime.now(timezone.utc) - account.created_at.replace(tzinfo=timezone.utc)).days
            features["account_age_days"] = max(age_days, 0)
        else:
            features["account_age_days"] = 0

        # Follower/following ratio
        following = account.following_count or 1
        features["follower_following_ratio"] = (account.followers_count or 0) / following

        # Tweet rate (tweets per day)
        if features["account_age_days"] > 0:
            features["tweet_rate"] = (account.tweet_count or 0) / features["account_age_days"]
        else:
            features["tweet_rate"] = 0

        # Raw counts
        features["followers_count"] = account.followers_count or 0
        features["following_count"] = account.following_count or 0
        features["tweet_count"] = account.tweet_count or 0

        # Verification
        features["is_verified"] = 1 if account.is_verified else 0

        return features

    def _extract_timing_features(self, tweets: list[Tweet]) -> dict:
        """Extract timing-based features from tweets."""
        features = {}

        if not tweets:
            return self._default_timing_features()

        # Get tweet timestamps
        timestamps = [
            t.created_at for t in tweets if t.created_at
        ]

        if not timestamps:
            return self._default_timing_features()

        # Convert to Iran local time and extract hours
        iran_hours = []
        for ts in timestamps:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            iran_time = ts + self.IRAN_TZ_OFFSET
            iran_hours.append(iran_time.hour)

        # Hour distribution entropy
        hour_counts = Counter(iran_hours)
        hour_dist = [hour_counts.get(h, 0) / len(iran_hours) for h in range(24)]
        features["hour_entropy"] = self._entropy(hour_dist)

        # Night posting ratio (2-6am Iran time)
        night_tweets = sum(1 for h in iran_hours if 2 <= h < 6)
        features["night_posting_ratio"] = night_tweets / len(iran_hours)

        # Weekend posting ratio (Friday/Saturday in Iran)
        weekdays = []
        for ts in timestamps:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            iran_time = ts + self.IRAN_TZ_OFFSET
            weekdays.append(iran_time.weekday())

        weekend_tweets = sum(1 for wd in weekdays if wd in [4, 5])  # Fri=4, Sat=5
        features["weekend_posting_ratio"] = weekend_tweets / len(weekdays) if weekdays else 0

        # Inter-tweet intervals
        sorted_ts = sorted(timestamps)
        if len(sorted_ts) > 1:
            intervals = []
            for i in range(1, len(sorted_ts)):
                diff = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds()
                intervals.append(diff)

            intervals_arr = np.array(intervals)
            features["interval_mean"] = float(np.mean(intervals_arr))
            features["interval_std"] = float(np.std(intervals_arr))
            features["interval_min"] = float(np.min(intervals_arr))
            features["interval_max"] = float(np.max(intervals_arr))

            # Interval entropy (binned)
            interval_hours = intervals_arr / 3600  # Convert to hours
            bins = np.histogram(interval_hours, bins=24, range=(0, 24))[0]
            if bins.sum() > 0:
                bin_dist = bins / bins.sum()
                features["interval_entropy"] = self._entropy(bin_dist.tolist())
            else:
                features["interval_entropy"] = 0
        else:
            features["interval_mean"] = 0
            features["interval_std"] = 0
            features["interval_min"] = 0
            features["interval_max"] = 0
            features["interval_entropy"] = 0

        # Periodicity score (simplified FFT-based)
        if len(iran_hours) >= 24:
            features["periodicity_score"] = self._calculate_periodicity(iran_hours)
        else:
            features["periodicity_score"] = 0

        return features

    def _extract_content_features(self, tweets: list[Tweet]) -> dict:
        """Extract content-based features from tweets."""
        features = {}

        if not tweets:
            return self._default_content_features()

        total = len(tweets)

        # Tweet type ratios
        retweets = sum(1 for t in tweets if t.is_retweet)
        replies = sum(1 for t in tweets if t.is_reply)
        quotes = sum(1 for t in tweets if t.is_quote)
        originals = total - retweets - replies - quotes

        features["retweet_ratio"] = retweets / total
        features["reply_ratio"] = replies / total
        features["quote_ratio"] = quotes / total
        features["original_ratio"] = max(0, originals) / total

        # Hashtag features
        all_hashtags = []
        hashtags_per_tweet = []
        for t in tweets:
            hashtags = t.hashtags or []
            all_hashtags.extend(hashtags)
            hashtags_per_tweet.append(len(hashtags))

        features["avg_hashtags_per_tweet"] = np.mean(hashtags_per_tweet) if hashtags_per_tweet else 0
        features["unique_hashtag_ratio"] = (
            len(set(all_hashtags)) / len(all_hashtags) if all_hashtags else 0
        )

        # Mention features
        mentions_per_tweet = [len(t.mentions or []) for t in tweets]
        features["avg_mentions_per_tweet"] = np.mean(mentions_per_tweet) if mentions_per_tweet else 0

        # URL features
        tweets_with_urls = sum(1 for t in tweets if t.urls)
        features["url_inclusion_rate"] = tweets_with_urls / total

        # Text length features
        text_lengths = [len(t.text or "") for t in tweets]
        features["avg_text_length"] = np.mean(text_lengths) if text_lengths else 0
        features["text_length_std"] = np.std(text_lengths) if text_lengths else 0

        # Duplicate content detection
        texts = [t.text for t in tweets if t.text]
        unique_texts = set(texts)
        features["duplicate_content_ratio"] = 1 - (len(unique_texts) / len(texts)) if texts else 0

        return features

    def _extract_engagement_features(self, tweets: list[Tweet]) -> dict:
        """Extract engagement-based features from tweets."""
        features = {}

        if not tweets:
            return self._default_engagement_features()

        retweet_counts = [t.retweet_count or 0 for t in tweets]
        like_counts = [t.like_count or 0 for t in tweets]
        reply_counts = [t.reply_count or 0 for t in tweets]

        features["avg_retweets_received"] = np.mean(retweet_counts)
        features["avg_likes_received"] = np.mean(like_counts)
        features["avg_replies_received"] = np.mean(reply_counts)

        # Engagement variance (bots often have very consistent low engagement)
        total_engagement = [r + l + rp for r, l, rp in zip(retweet_counts, like_counts, reply_counts)]
        features["engagement_variance"] = np.var(total_engagement) if total_engagement else 0

        # Zero engagement ratio
        zero_engagement = sum(1 for e in total_engagement if e == 0)
        features["zero_engagement_ratio"] = zero_engagement / len(total_engagement) if total_engagement else 0

        return features

    def _default_timing_features(self) -> dict:
        """Default timing features when no tweets available."""
        return {
            "hour_entropy": 0,
            "night_posting_ratio": 0,
            "weekend_posting_ratio": 0,
            "interval_mean": 0,
            "interval_std": 0,
            "interval_min": 0,
            "interval_max": 0,
            "interval_entropy": 0,
            "periodicity_score": 0,
        }

    def _default_content_features(self) -> dict:
        """Default content features when no tweets available."""
        return {
            "retweet_ratio": 0,
            "reply_ratio": 0,
            "quote_ratio": 0,
            "original_ratio": 0,
            "avg_hashtags_per_tweet": 0,
            "unique_hashtag_ratio": 0,
            "avg_mentions_per_tweet": 0,
            "url_inclusion_rate": 0,
            "avg_text_length": 0,
            "text_length_std": 0,
            "duplicate_content_ratio": 0,
        }

    def _default_engagement_features(self) -> dict:
        """Default engagement features when no tweets available."""
        return {
            "avg_retweets_received": 0,
            "avg_likes_received": 0,
            "avg_replies_received": 0,
            "engagement_variance": 0,
            "zero_engagement_ratio": 0,
        }

    def _entropy(self, distribution: list[float]) -> float:
        """Calculate Shannon entropy of a distribution."""
        entropy = 0
        for p in distribution:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _calculate_periodicity(self, hours: list[int]) -> float:
        """Calculate periodicity score using FFT.

        Higher score indicates more periodic/automated posting pattern.
        """
        if len(hours) < 24:
            return 0

        # Create hour histogram
        hour_counts = Counter(hours)
        signal = [hour_counts.get(h, 0) for h in range(24)]

        # Apply FFT
        fft_result = np.fft.fft(signal)
        magnitudes = np.abs(fft_result)[1:13]  # Skip DC, take first half

        # Periodicity score: ratio of dominant frequency to total
        if magnitudes.sum() > 0:
            return float(magnitudes.max() / magnitudes.sum())
        return 0

    def extract_features_for_account(self, account_id: int) -> Optional[dict]:
        """Extract and store features for an account by ID."""
        account = self.db.get_account(account_id)
        if not account:
            return None

        tweets = self.db.get_tweets_by_account(account_id)
        features = self.extract_account_features(account, tweets)

        # Store features
        self.db.update_account_features(account_id, features)

        return features

    def extract_features_batch(self, account_ids: Optional[list[int]] = None) -> int:
        """Extract features for multiple accounts.

        Args:
            account_ids: List of account IDs. If None, process all accounts.

        Returns:
            Number of accounts processed.
        """
        if account_ids is None:
            accounts = self.db.get_all_accounts()
            account_ids = [a.id for a in accounts if a.id]

        processed = 0
        for account_id in account_ids:
            if self.extract_features_for_account(account_id):
                processed += 1

        return processed


# Feature names for model training
FEATURE_NAMES = [
    # Profile features
    "username_digit_ratio",
    "username_length",
    "bio_length",
    "has_bio",
    "has_default_avatar",
    "account_age_days",
    "follower_following_ratio",
    "tweet_rate",
    "followers_count",
    "following_count",
    "tweet_count",
    "is_verified",
    # Timing features
    "hour_entropy",
    "night_posting_ratio",
    "weekend_posting_ratio",
    "interval_mean",
    "interval_std",
    "interval_min",
    "interval_max",
    "interval_entropy",
    "periodicity_score",
    # Content features
    "retweet_ratio",
    "reply_ratio",
    "quote_ratio",
    "original_ratio",
    "avg_hashtags_per_tweet",
    "unique_hashtag_ratio",
    "avg_mentions_per_tweet",
    "url_inclusion_rate",
    "avg_text_length",
    "text_length_std",
    "duplicate_content_ratio",
    # Engagement features
    "avg_retweets_received",
    "avg_likes_received",
    "avg_replies_received",
    "engagement_variance",
    "zero_engagement_ratio",
]


def features_to_vector(features: dict) -> list[float]:
    """Convert features dict to ordered vector for model input."""
    return [features.get(name, 0) for name in FEATURE_NAMES]
