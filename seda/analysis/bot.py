"""Bot detection for SEDA."""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from seda.config import get_settings
from seda.db import get_db
from seda.models import Account
from seda.analysis.features import FeatureExtractor, features_to_vector, FEATURE_NAMES

# Try to import lightgbm, fallback to heuristics only
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BotDetector:
    """Bot detection using LightGBM classifier and heuristics."""

    def __init__(self):
        self.db = get_db()
        self.feature_extractor = FeatureExtractor()
        self.model: Optional[lgb.Booster] = None
        self.settings = get_settings()

        # Try to load existing model
        self._load_model()

    def _load_model(self) -> bool:
        """Load trained model if available."""
        if not LIGHTGBM_AVAILABLE:
            return False

        model_path = self.settings.models_path / "bot_classifier.lgb"
        if model_path.exists():
            try:
                self.model = lgb.Booster(model_file=str(model_path))
                return True
            except Exception:
                pass
        return False

    def _save_model(self, model: lgb.Booster) -> None:
        """Save trained model."""
        model_path = self.settings.models_path / "bot_classifier.lgb"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))

    def score_account(self, account: Account, features: Optional[dict] = None) -> float:
        """Score an account for bot probability.

        Args:
            account: Account to score
            features: Pre-computed features (optional)

        Returns:
            Bot probability from 0 (human) to 1 (bot)
        """
        # Get features if not provided
        if features is None and account.features:
            features = account.features
        elif features is None and account.id:
            tweets = self.db.get_tweets_by_account(account.id)
            features = self.feature_extractor.extract_account_features(account, tweets)

        if features is None:
            return 0.5  # Unknown

        # Use model if available
        if self.model and LIGHTGBM_AVAILABLE:
            try:
                feature_vector = np.array([features_to_vector(features)])
                score = self.model.predict(feature_vector)[0]
                return float(score)
            except Exception:
                pass

        # Fallback to heuristics
        return self._heuristic_score(account, features)

    def _heuristic_score(self, account: Account, features: dict) -> float:
        """Calculate bot score using heuristics when model unavailable.

        Returns score from 0 (likely human) to 1 (likely bot).
        """
        score = 0.0
        weights_sum = 0.0

        # Username digit ratio (bots often have many digits)
        digit_ratio = features.get("username_digit_ratio", 0)
        if digit_ratio > 0.5:
            score += 0.3
        elif digit_ratio > 0.3:
            score += 0.15
        weights_sum += 0.3

        # Bio presence (lack of bio is suspicious)
        if not features.get("has_bio", 1):
            score += 0.1
        weights_sum += 0.1

        # Default avatar
        if features.get("has_default_avatar", 0):
            score += 0.15
        weights_sum += 0.15

        # Account age (very new accounts are suspicious)
        age_days = features.get("account_age_days", 365)
        if age_days < 30:
            score += 0.2
        elif age_days < 90:
            score += 0.1
        weights_sum += 0.2

        # Tweet rate (very high rate is suspicious)
        tweet_rate = features.get("tweet_rate", 0)
        if tweet_rate > 100:  # More than 100 tweets/day
            score += 0.25
        elif tweet_rate > 50:
            score += 0.15
        weights_sum += 0.25

        # Follower ratio (very low is suspicious)
        ratio = features.get("follower_following_ratio", 1)
        if ratio < 0.01:
            score += 0.1
        weights_sum += 0.1

        # Hour entropy (low entropy = posting at same times = suspicious)
        hour_entropy = features.get("hour_entropy", 3)
        if hour_entropy < 2:
            score += 0.2
        elif hour_entropy < 3:
            score += 0.1
        weights_sum += 0.2

        # Night posting ratio (posting 2-6am local time is suspicious)
        night_ratio = features.get("night_posting_ratio", 0)
        if night_ratio > 0.3:
            score += 0.15
        weights_sum += 0.15

        # Retweet ratio (very high = amplifier bot)
        retweet_ratio = features.get("retweet_ratio", 0)
        if retweet_ratio > 0.9:
            score += 0.2
        elif retweet_ratio > 0.7:
            score += 0.1
        weights_sum += 0.2

        # Duplicate content
        duplicate_ratio = features.get("duplicate_content_ratio", 0)
        if duplicate_ratio > 0.5:
            score += 0.2
        weights_sum += 0.2

        # Zero engagement ratio (always zero engagement is suspicious)
        zero_engagement = features.get("zero_engagement_ratio", 0)
        if zero_engagement > 0.9:
            score += 0.15
        weights_sum += 0.15

        # Periodicity score (high periodicity = automated)
        periodicity = features.get("periodicity_score", 0)
        if periodicity > 0.5:
            score += 0.15
        weights_sum += 0.15

        # Normalize to 0-1
        return min(1.0, score / weights_sum) if weights_sum > 0 else 0.5

    def score_all_accounts(self, account_ids: Optional[list[int]] = None) -> int:
        """Score all accounts (or specified accounts) for bot probability.

        Returns number of accounts scored.
        """
        if account_ids is None:
            accounts = self.db.get_all_accounts()
        else:
            accounts = [self.db.get_account(aid) for aid in account_ids]
            accounts = [a for a in accounts if a]

        scored = 0
        for account in accounts:
            if not account.id:
                continue

            # Get or extract features
            if not account.features:
                features = self.feature_extractor.extract_features_for_account(account.id)
            else:
                features = account.features

            if features:
                score = self.score_account(account, features)
                self.db.update_account_scores(account.id, bot_score=score)
                scored += 1

        return scored

    def train_model(
        self,
        positive_ids: list[int],
        negative_ids: list[int],
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        """Train a LightGBM bot detection model.

        Args:
            positive_ids: Account IDs labeled as bots
            negative_ids: Account IDs labeled as humans
            params: LightGBM parameters (optional)

        Returns:
            Training metrics or None if training failed
        """
        if not LIGHTGBM_AVAILABLE:
            return None

        # Default parameters
        if params is None:
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
            }

        # Collect features
        X = []
        y = []

        for account_id in positive_ids:
            account = self.db.get_account(account_id)
            if account and account.features:
                X.append(features_to_vector(account.features))
                y.append(1)

        for account_id in negative_ids:
            account = self.db.get_account(account_id)
            if account and account.features:
                X.append(features_to_vector(account.features))
                y.append(0)

        if len(X) < 10:
            return None  # Not enough data

        X = np.array(X)
        y = np.array(y)

        # Create dataset
        train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
        )

        # Save model
        self._save_model(self.model)

        # Return metrics
        preds = self.model.predict(X)
        from sklearn.metrics import roc_auc_score, accuracy_score

        return {
            "auc": roc_auc_score(y, preds),
            "accuracy": accuracy_score(y, (preds > 0.5).astype(int)),
            "num_samples": len(y),
            "num_positive": sum(y),
            "num_negative": len(y) - sum(y),
        }

    def get_high_confidence_bots(
        self, min_score: float = 0.8, limit: int = 100
    ) -> list[Account]:
        """Get accounts with high bot scores."""
        return self.db.get_all_accounts(
            min_bot_score=min_score, limit=limit
        )

    def get_high_confidence_humans(
        self, max_score: float = 0.2, limit: int = 100
    ) -> list[Account]:
        """Get accounts with low bot scores (likely humans)."""
        return self.db.get_all_accounts(
            max_bot_score=max_score, limit=limit
        )
