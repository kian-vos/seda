"""Coordination detection for SEDA."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from seda.config import get_settings
from seda.db import get_db
from seda.models import (
    Account,
    CoordinationCluster,
    CoordinationType,
    PoliticalStance,
    Tweet,
)

# Try to import dependencies
try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

try:
    import networkx as nx
    from community import community_louvain
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class CoordinationDetector:
    """Detect coordinated inauthentic behavior."""

    def __init__(self):
        self.db = get_db()
        self.settings = get_settings()

    def detect_all(self) -> dict:
        """Run all coordination detection algorithms.

        Returns dict with counts of clusters found by each method.
        """
        results = {
            "synchronized_posting": 0,
            "hashtag_campaigns": 0,
            "amplification_networks": 0,
        }

        # Synchronized posting (near-duplicate tweets)
        if DATASKETCH_AVAILABLE:
            clusters = self.detect_synchronized_posting()
            results["synchronized_posting"] = len(clusters)

        # Hashtag velocity spikes
        campaigns = self.detect_hashtag_campaigns()
        results["hashtag_campaigns"] = len(campaigns)

        # Amplification networks
        if NETWORKX_AVAILABLE:
            networks = self.detect_amplification_networks()
            results["amplification_networks"] = len(networks)

        return results

    def detect_synchronized_posting(
        self,
        time_window_minutes: Optional[int] = None,
        min_cluster_size: int = 20,  # Require 20+ accounts for real coordination
    ) -> list[CoordinationCluster]:
        """Detect synchronized posting using MinHash LSH.

        Finds near-duplicate tweets posted within a time window.
        """
        if not DATASKETCH_AVAILABLE:
            return []

        time_window = time_window_minutes or self.settings.coordination_time_window_minutes
        threshold = self.settings.minhash_threshold
        num_perm = self.settings.minhash_num_perm

        # Get all tweets
        tweets = self.db.get_all_tweets()
        if len(tweets) < min_cluster_size:
            return []

        # Create MinHash for each tweet
        minhashes = {}
        tweet_map = {}

        for tweet in tweets:
            if not tweet.text or not tweet.id:
                continue

            # Create shingles (word n-grams)
            words = tweet.text.lower().split()
            if len(words) < 3:
                continue

            shingles = set()
            for i in range(len(words) - 2):
                shingle = " ".join(words[i : i + 3])
                shingles.add(shingle)

            if not shingles:
                continue

            # Create MinHash
            mh = MinHash(num_perm=num_perm)
            for shingle in shingles:
                mh.update(shingle.encode("utf-8"))

            minhashes[tweet.id] = mh
            tweet_map[tweet.id] = tweet

        if len(minhashes) < min_cluster_size:
            return []

        # Create LSH index
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for tweet_id, mh in minhashes.items():
            try:
                lsh.insert(str(tweet_id), mh)
            except ValueError:
                pass  # Duplicate key

        # Find clusters
        processed = set()
        clusters = []

        for tweet_id, mh in minhashes.items():
            if tweet_id in processed:
                continue

            # Find similar tweets
            similar_ids = [int(sid) for sid in lsh.query(mh)]
            if len(similar_ids) < min_cluster_size:
                continue

            # Filter by time window
            base_tweet = tweet_map[tweet_id]
            if not base_tweet.created_at:
                continue

            time_window_delta = timedelta(minutes=time_window)
            coordinated_tweets = []

            for sid in similar_ids:
                similar_tweet = tweet_map.get(sid)
                if not similar_tweet or not similar_tweet.created_at:
                    continue

                time_diff = abs(
                    (similar_tweet.created_at - base_tweet.created_at).total_seconds()
                )
                if time_diff <= time_window_delta.total_seconds():
                    coordinated_tweets.append(similar_tweet)
                    processed.add(sid)

            if len(coordinated_tweets) >= min_cluster_size:
                # Get unique accounts
                account_ids = list(set(t.account_id for t in coordinated_tweets))

                if len(account_ids) >= min_cluster_size:
                    cluster = CoordinationCluster(
                        cluster_type=CoordinationType.SYNCHRONIZED_POSTING,
                        member_account_ids=account_ids,
                        evidence={
                            "sample_text": base_tweet.text[:200],
                            "tweet_count": len(coordinated_tweets),
                            "time_span_minutes": time_window,
                        },
                        confidence_score=len(account_ids) / 10,  # Scale by size
                        description=f"Synchronized posting: {len(account_ids)} accounts posted similar content within {time_window} minutes",
                    )
                    cluster_id = self.db.insert_cluster(cluster)
                    cluster.id = cluster_id
                    clusters.append(cluster)

        return clusters

    def detect_hashtag_campaigns(
        self,
        z_score_threshold: float = 3.0,
        min_accounts: int = 50,  # Require 50+ UNIQUE accounts for real coordination
    ) -> list[CoordinationCluster]:
        """Detect coordinated hashtag campaigns.

        Finds hashtags with sudden spikes driven by many coordinated accounts.
        Real coordination = many different accounts posting same hashtag simultaneously.
        """
        # Get all tweets with hashtags
        tweets = self.db.get_all_tweets()

        # Group tweets by hashtag and time bucket (1 hour)
        hashtag_timeline: dict[str, dict[str, list[Tweet]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for tweet in tweets:
            if not tweet.hashtags or not tweet.created_at:
                continue

            hour_bucket = tweet.created_at.strftime("%Y-%m-%d-%H")

            for hashtag in tweet.hashtags:
                hashtag_lower = hashtag.lower()
                hashtag_timeline[hashtag_lower][hour_bucket].append(tweet)

        clusters = []

        for hashtag, timeline in hashtag_timeline.items():
            if len(timeline) < 3:  # Need some history
                continue

            # Calculate hourly counts
            bucket_counts = [len(tweets) for tweets in timeline.values()]

            if len(bucket_counts) < 3:
                continue

            mean_count = np.mean(bucket_counts)
            std_count = np.std(bucket_counts)

            if std_count == 0:
                continue

            # Find spike buckets
            for bucket, bucket_tweets in timeline.items():
                count = len(bucket_tweets)
                z_score = (count - mean_count) / std_count

                if z_score >= z_score_threshold:
                    # Check if spike is driven by MANY coordinated accounts
                    account_ids = list(set(t.account_id for t in bucket_tweets))
                    unique_account_count = len(account_ids)

                    # Real coordination = many different accounts, not one account spamming
                    # Require: 50+ unique accounts AND most accounts only tweeted 1-2 times
                    # (organic spam is 1 account posting 100 times, coordination is 100 accounts posting 1-2 times each)
                    if unique_account_count >= min_accounts:
                        avg_tweets_per_account = count / unique_account_count

                        # Coordination signature: many accounts, low tweets per account
                        if avg_tweets_per_account <= 3:  # Each account tweeting 1-3 times max
                            cluster = CoordinationCluster(
                                cluster_type=CoordinationType.HASHTAG_CAMPAIGN,
                                member_account_ids=account_ids,
                                evidence={
                                    "hashtag": hashtag,
                                    "time_bucket": bucket,
                                    "tweet_count": count,
                                    "unique_accounts": unique_account_count,
                                    "z_score": float(z_score),
                                    "avg_tweets_per_account": float(avg_tweets_per_account),
                                },
                                confidence_score=min(1.0, unique_account_count / 100),  # Scale by account count
                                description=f"Hashtag campaign: #{hashtag} - {unique_account_count} accounts posted {count} tweets in 1 hour (z={z_score:.1f})",
                            )
                            cluster_id = self.db.insert_cluster(cluster)
                            cluster.id = cluster_id
                            clusters.append(cluster)

        return clusters

    def detect_amplification_networks(
        self,
        min_retweets_to_seeds: int = 5,
        min_community_size: int = 20,  # Require 20+ accounts for real coordination
    ) -> list[CoordinationCluster]:
        """Detect amplification networks using retweet graph analysis.

        Finds clusters of accounts that coordinate to amplify seed accounts.
        """
        if not NETWORKX_AVAILABLE:
            return []

        # Get seed accounts
        seeds = self.db.get_all_accounts(is_seed=True)
        seed_ids = {s.id for s in seeds if s.id}

        if not seed_ids:
            return []

        # Build retweet graph
        # Edge from A to B means A retweeted B
        retweet_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        tweets = self.db.get_all_tweets()

        for tweet in tweets:
            if not tweet.is_retweet or not tweet.referenced_tweet_id:
                continue

            # Find the original tweet author
            # Note: We might not have the original tweet, so we track by account_id
            retweeter_id = tweet.account_id

            # Check mentions for original author
            for mention in tweet.mentions or []:
                # Try to find mentioned account
                mentioned = self.db.get_account_by_username(mention.lower())
                if mentioned and mentioned.id:
                    retweet_counts[retweeter_id][mentioned.id] += 1

        # Find high-amplifiers (accounts that retweet seeds frequently)
        amplifiers = {}
        for retweeter_id, targets in retweet_counts.items():
            seed_retweets = sum(
                count for target_id, count in targets.items() if target_id in seed_ids
            )
            if seed_retweets >= min_retweets_to_seeds:
                amplifiers[retweeter_id] = seed_retweets

        if len(amplifiers) < min_community_size:
            return []

        # Build co-amplification graph
        # Edge between A and B if they both retweet the same seed accounts
        G = nx.Graph()

        amplifier_ids = list(amplifiers.keys())
        G.add_nodes_from(amplifier_ids)

        for i, amp1 in enumerate(amplifier_ids):
            targets1 = set(retweet_counts[amp1].keys()) & seed_ids
            for amp2 in amplifier_ids[i + 1 :]:
                targets2 = set(retweet_counts[amp2].keys()) & seed_ids
                overlap = len(targets1 & targets2)
                if overlap > 0:
                    G.add_edge(amp1, amp2, weight=overlap)

        # Run community detection
        if G.number_of_edges() == 0:
            return []

        try:
            communities = community_louvain.best_partition(G)
        except Exception:
            return []

        # Group accounts by community
        community_members: dict[int, list[int]] = defaultdict(list)
        for account_id, community_id in communities.items():
            community_members[community_id].append(account_id)

        clusters = []

        for community_id, member_ids in community_members.items():
            if len(member_ids) < min_community_size:
                continue

            # Calculate total amplification
            total_retweets = sum(amplifiers.get(mid, 0) for mid in member_ids)

            # Get common targets
            common_targets: dict[int, int] = defaultdict(int)
            for mid in member_ids:
                for target_id, count in retweet_counts[mid].items():
                    if target_id in seed_ids:
                        common_targets[target_id] += count

            top_targets = sorted(common_targets.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]

            cluster = CoordinationCluster(
                cluster_type=CoordinationType.AMPLIFICATION_NETWORK,
                member_account_ids=member_ids,
                evidence={
                    "community_id": community_id,
                    "total_seed_retweets": total_retweets,
                    "top_amplified_accounts": [
                        {"account_id": tid, "retweet_count": cnt}
                        for tid, cnt in top_targets
                    ],
                },
                confidence_score=min(1.0, len(member_ids) / 20),
                description=f"Amplification network: {len(member_ids)} accounts coordinating to amplify seed accounts",
            )
            cluster_id = self.db.insert_cluster(cluster)
            cluster.id = cluster_id
            clusters.append(cluster)

        return clusters

    def get_account_coordination_score(self, account_id: int) -> float:
        """Calculate coordination score for an account.

        Score based on participation in coordination clusters.
        """
        clusters = self.db.get_clusters()

        involvement_count = 0
        total_confidence = 0.0

        for cluster in clusters:
            if account_id in cluster.member_account_ids:
                involvement_count += 1
                total_confidence += cluster.confidence_score

        if involvement_count == 0:
            return 0.0

        # Score increases with cluster involvement
        return min(1.0, total_confidence / 3)

    def update_coordination_scores(self) -> int:
        """Update coordination scores for all accounts in clusters."""
        clusters = self.db.get_clusters()

        # Collect all account IDs involved in clusters
        involved_accounts: set[int] = set()
        for cluster in clusters:
            involved_accounts.update(cluster.member_account_ids)

        updated = 0
        for account_id in involved_accounts:
            score = self.get_account_coordination_score(account_id)
            self.db.update_account_scores(account_id, coordination_score=score)
            updated += 1

        return updated

    def get_cluster_details(self, cluster_id: int) -> Optional[dict]:
        """Get detailed information about a cluster."""
        clusters = self.db.get_clusters()
        cluster = next((c for c in clusters if c.id == cluster_id), None)

        if not cluster:
            return None

        # Get account details
        accounts = []
        for account_id in cluster.member_account_ids:
            account = self.db.get_account(account_id)
            if account:
                accounts.append(
                    {
                        "id": account.id,
                        "username": account.username,
                        "display_name": account.display_name,
                        "bot_score": account.bot_score,
                        "stance": account.political_stance.value if account.political_stance else None,
                    }
                )

        return {
            "cluster": {
                "id": cluster.id,
                "type": cluster.cluster_type.value,
                "confidence": cluster.confidence_score,
                "description": cluster.description,
                "created_at": cluster.created_at.isoformat() if cluster.created_at else None,
            },
            "evidence": cluster.evidence,
            "accounts": accounts,
        }
