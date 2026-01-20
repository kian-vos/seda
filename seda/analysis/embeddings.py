"""ParsBERT embeddings for semantic analysis in SEDA."""

from typing import Optional

import numpy as np

from seda.config import get_settings
from seda.db import get_db
from seda.models import Account, Tweet

# Try to import transformers, fallback if not available
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import sklearn for similarity calculations
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PersianEmbedder:
    """Persian text embeddings using ParsBERT.

    ParsBERT (HooshvareLab/bert-fa-base-uncased) is a BERT model pre-trained
    specifically on Persian text. It provides 768-dimensional embeddings that
    capture semantic meaning.

    Benefits over keyword matching:
    - Semantic meaning: Similar concepts have similar embeddings
    - Context sensitivity: Same word means different things in different contexts
    - Sarcasm/irony detection: Better at detecting tone
    - Coded language: Can detect euphemisms and slang
    """

    MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
    EMBEDDING_DIM = 768

    def __init__(self, device: Optional[str] = None):
        """Initialize the Persian embedder.

        Args:
            device: Device to run model on ('cpu', 'cuda', 'mps'). Auto-detects if None.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for ParsBERT embeddings. "
                "Install with: pip install transformers torch"
            )

        self.settings = get_settings()
        self.db = get_db()

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        """Get 768-dimensional embedding for text.

        Args:
            text: Text to embed

        Returns:
            768-dimensional numpy array
        """
        if not text:
            return np.zeros(self.EMBEDDING_DIM)

        # Tokenize with truncation
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy().flatten()

    def embed_texts_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            Array of shape (n_texts, 768)
        """
        if not texts:
            return np.array([])

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0 to 1)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is required for similarity calculations. "
                "Install with: pip install scikit-learn"
            )

        emb1 = self.embed_text(text1).reshape(1, -1)
        emb2 = self.embed_text(text2).reshape(1, -1)

        return float(cosine_similarity(emb1, emb2)[0][0])

    def find_similar_texts(
        self,
        query_text: str,
        candidate_texts: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Find most similar texts to a query.

        Args:
            query_text: Text to find matches for
            candidate_texts: List of texts to search
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is required for similarity search. "
                "Install with: pip install scikit-learn"
            )

        if not candidate_texts:
            return []

        # Embed query
        query_emb = self.embed_text(query_text).reshape(1, -1)

        # Embed candidates
        candidate_embs = self.embed_texts_batch(candidate_texts)

        # Calculate similarities
        similarities = cosine_similarity(query_emb, candidate_embs)[0]

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def embed_account(self, account: Account, save: bool = True) -> np.ndarray:
        """Create embedding for an account based on bio and recent tweets.

        Args:
            account: Account to embed
            save: Whether to save embedding to database

        Returns:
            768-dimensional embedding
        """
        # Combine bio and recent tweets
        texts = []
        if account.bio:
            texts.append(account.bio)

        if account.id:
            tweets = self.db.get_tweets_by_account(account.id, limit=20)
            for tweet in tweets:
                if tweet.text and not tweet.is_retweet:
                    texts.append(tweet.text)

        if not texts:
            return np.zeros(self.EMBEDDING_DIM)

        # Combine all text
        combined_text = " ".join(texts)
        embedding = self.embed_text(combined_text)

        # Save to database
        if save and account.id:
            self.db.update_account_embedding(account.id, embedding.tolist())

        return embedding

    def embed_tweet(self, tweet: Tweet, save: bool = True) -> np.ndarray:
        """Create embedding for a tweet.

        Args:
            tweet: Tweet to embed
            save: Whether to save embedding to database

        Returns:
            768-dimensional embedding
        """
        if not tweet.text:
            return np.zeros(self.EMBEDDING_DIM)

        embedding = self.embed_text(tweet.text)

        # Save to database
        if save and tweet.id:
            self.db.update_tweet_embedding(tweet.id, embedding.tolist())

        return embedding

    def embed_all_accounts(
        self,
        account_ids: Optional[list[int]] = None,
        batch_size: int = 32,
    ) -> int:
        """Embed all accounts (or specified accounts).

        Args:
            account_ids: Optional list of account IDs to embed
            batch_size: Batch size for processing

        Returns:
            Number of accounts embedded
        """
        if account_ids is None:
            accounts = self.db.get_all_accounts()
        else:
            accounts = [self.db.get_account(aid) for aid in account_ids]
            accounts = [a for a in accounts if a]

        embedded = 0
        for account in accounts:
            if account.id:
                self.embed_account(account, save=True)
                embedded += 1

        return embedded

    def embed_all_tweets(
        self,
        tweet_ids: Optional[list[int]] = None,
        limit: int = 10000,
    ) -> int:
        """Embed all tweets (or specified tweets).

        Args:
            tweet_ids: Optional list of tweet IDs to embed
            limit: Maximum number of tweets to process

        Returns:
            Number of tweets embedded
        """
        if tweet_ids is None:
            tweets = self.db.get_all_tweets(limit=limit)
        else:
            tweets = [self.db.get_tweet(tid) for tid in tweet_ids]
            tweets = [t for t in tweets if t]

        embedded = 0
        for tweet in tweets:
            if tweet.id:
                self.embed_tweet(tweet, save=True)
                embedded += 1

        return embedded

    def find_similar_accounts(
        self,
        account: Account,
        top_k: int = 10,
    ) -> list[tuple[Account, float]]:
        """Find accounts most similar to a given account.

        Args:
            account: Account to find matches for
            top_k: Number of results to return

        Returns:
            List of (Account, similarity_score) tuples
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is required for similarity search. "
                "Install with: pip install scikit-learn"
            )

        # Get query embedding
        if account.embedding:
            query_emb = np.array(account.embedding).reshape(1, -1)
        else:
            query_emb = self.embed_account(account, save=False).reshape(1, -1)

        # Get all accounts with embeddings
        all_accounts = self.db.get_all_accounts()
        accounts_with_emb = [a for a in all_accounts if a.embedding and a.id != account.id]

        if not accounts_with_emb:
            return []

        # Stack embeddings
        candidate_embs = np.array([a.embedding for a in accounts_with_emb])

        # Calculate similarities
        similarities = cosine_similarity(query_emb, candidate_embs)[0]

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (accounts_with_emb[idx], float(similarities[idx]))
            for idx in top_indices
        ]

    def cluster_accounts_by_content(
        self,
        accounts: list[Account],
        n_clusters: int = 10,
    ) -> dict[int, list[Account]]:
        """Cluster accounts by their content similarity.

        Args:
            accounts: Accounts to cluster
            n_clusters: Number of clusters

        Returns:
            Dict mapping cluster ID to list of accounts
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "sklearn is required for clustering. "
                "Install with: pip install scikit-learn"
            )

        # Filter accounts with embeddings
        accounts_with_emb = [a for a in accounts if a.embedding]

        if len(accounts_with_emb) < n_clusters:
            # Not enough accounts for requested clusters
            return {0: accounts_with_emb}

        # Stack embeddings
        embeddings = np.array([a.embedding for a in accounts_with_emb])

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group by cluster
        clusters: dict[int, list[Account]] = {}
        for account, label in zip(accounts_with_emb, cluster_labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(account)

        return clusters


# Singleton instance
_embedder: Optional[PersianEmbedder] = None


def get_embedder() -> PersianEmbedder:
    """Get or create global PersianEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = PersianEmbedder()
    return _embedder


def is_embeddings_available() -> bool:
    """Check if embeddings functionality is available."""
    return TRANSFORMERS_AVAILABLE
