"""Database operations for SEDA using SQLite with FTS5 or Turso."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, Any

from seda.config import get_settings

# Try to import libsql for Turso support
try:
    import libsql_experimental as libsql
    LIBSQL_AVAILABLE = True
except ImportError:
    LIBSQL_AVAILABLE = False
from seda.models import (
    Account,
    AccountType,
    CoordinationCluster,
    CoordinationType,
    PoliticalStance,
    PoliticalTaxonomy,
    SeedAccount,
    SeedCategory,
    ThreatLevel,
    Tweet,
)

SCHEMA = """
-- Accounts table
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    twitter_id TEXT UNIQUE NOT NULL,
    username TEXT NOT NULL,
    display_name TEXT DEFAULT '',
    bio TEXT DEFAULT '',
    followers_count INTEGER DEFAULT 0,
    following_count INTEGER DEFAULT 0,
    tweet_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    profile_image_url TEXT DEFAULT '',
    is_verified BOOLEAN DEFAULT FALSE,
    location TEXT DEFAULT '',
    bot_score REAL,
    regime_score REAL,
    coordination_score REAL,
    account_type TEXT DEFAULT 'unknown',
    political_stance TEXT DEFAULT 'unknown',
    political_taxonomy TEXT DEFAULT 'unknown',
    threat_level TEXT DEFAULT 'unknown',
    is_seed BOOLEAN DEFAULT FALSE,
    seed_category TEXT,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    features TEXT,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_accounts_username ON accounts(username);
CREATE INDEX IF NOT EXISTS idx_accounts_bot_score ON accounts(bot_score);
CREATE INDEX IF NOT EXISTS idx_accounts_political_stance ON accounts(political_stance);
CREATE INDEX IF NOT EXISTS idx_accounts_threat_level ON accounts(threat_level);
CREATE INDEX IF NOT EXISTS idx_accounts_is_seed ON accounts(is_seed);

-- Tweets table
CREATE TABLE IF NOT EXISTS tweets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id TEXT UNIQUE NOT NULL,
    account_id INTEGER NOT NULL,
    text TEXT DEFAULT '',
    created_at TIMESTAMP,
    hashtags TEXT DEFAULT '[]',
    mentions TEXT DEFAULT '[]',
    urls TEXT DEFAULT '[]',
    retweet_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    is_retweet BOOLEAN DEFAULT FALSE,
    is_reply BOOLEAN DEFAULT FALSE,
    is_quote BOOLEAN DEFAULT FALSE,
    referenced_tweet_id TEXT,
    sentiment REAL,
    regime_alignment REAL,
    talking_points TEXT DEFAULT '[]',
    embedding BLOB,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(id)
);

CREATE INDEX IF NOT EXISTS idx_tweets_account_id ON tweets(account_id);
CREATE INDEX IF NOT EXISTS idx_tweets_created_at ON tweets(created_at);
CREATE INDEX IF NOT EXISTS idx_tweets_is_retweet ON tweets(is_retweet);

-- Coordination clusters table
CREATE TABLE IF NOT EXISTS coordination_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    member_account_ids TEXT DEFAULT '[]',
    evidence TEXT DEFAULT '{}',
    confidence_score REAL DEFAULT 0.0,
    description TEXT DEFAULT '',
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_clusters_type ON coordination_clusters(cluster_type);
CREATE INDEX IF NOT EXISTS idx_clusters_confidence ON coordination_clusters(confidence_score);

-- Seed accounts table
CREATE TABLE IF NOT EXISTS seed_accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT DEFAULT ''
);

-- FTS5 virtual table for tweet text search
CREATE VIRTUAL TABLE IF NOT EXISTS tweets_fts USING fts5(
    text,
    content='tweets',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS tweets_ai AFTER INSERT ON tweets BEGIN
    INSERT INTO tweets_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS tweets_ad AFTER DELETE ON tweets BEGIN
    INSERT INTO tweets_fts(tweets_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS tweets_au AFTER UPDATE ON tweets BEGIN
    INSERT INTO tweets_fts(tweets_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO tweets_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


class DictRow(dict):
    """Dict that also supports index access like sqlite3.Row."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class TursoCursorWrapper:
    """Wrapper to make Turso cursor return dict-like rows."""

    def __init__(self, cursor):
        self._cursor = cursor
        self._description = cursor.description if hasattr(cursor, 'description') else None

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def fetchall(self):
        rows = self._cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row):
        if self._description:
            columns = [col[0] for col in self._description]
            return DictRow(dict(zip(columns, row)))
        return DictRow(dict(enumerate(row)))

    @property
    def rowcount(self):
        return getattr(self._cursor, 'rowcount', -1)


class TursoConnectionWrapper:
    """Wrapper to make Turso connection behave like sqlite3 with Row factory."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params = None) -> TursoCursorWrapper:
        if params:
            # libsql expects positional args, convert to tuple if needed
            if isinstance(params, (list, tuple)):
                cursor = self._conn.execute(sql, tuple(params))
            else:
                cursor = self._conn.execute(sql, (params,))
        else:
            cursor = self._conn.execute(sql)
        return TursoCursorWrapper(cursor)

    def executemany(self, sql: str, params_list: list) -> None:
        for params in params_list:
            self._conn.execute(sql, tuple(params))

    def executescript(self, script: str) -> None:
        for statement in script.split(';'):
            statement = statement.strip()
            if statement:
                self._conn.execute(statement)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()


class Database:
    """Database operations for SEDA."""

    def __init__(self, db_path: Optional[Path] = None):
        self.settings = get_settings()
        self.use_turso = self.settings.use_turso and LIBSQL_AVAILABLE

        if self.use_turso:
            # Use Turso cloud database
            self.turso_url = self.settings.turso_database_url
            self.turso_token = self.settings.turso_auth_token
            self.db_path = None
        else:
            # Use local SQLite
            self.db_path = db_path or self.settings.db_path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.turso_url = None
            self.turso_token = None

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self.connection() as conn:
            # For Turso, execute statements one by one (no executescript)
            if self.use_turso:
                for statement in SCHEMA.split(';'):
                    statement = statement.strip()
                    if statement and not statement.startswith('--'):
                        # Skip FTS5 and triggers for Turso (not fully supported)
                        if 'FTS5' in statement.upper() or 'TRIGGER' in statement.upper():
                            continue
                        try:
                            conn.execute(statement)
                        except Exception:
                            pass  # Ignore errors for CREATE IF NOT EXISTS
                conn.commit()
                # Run migrations to add new columns to existing tables
                self._run_migrations(conn)
            else:
                conn.executescript(SCHEMA)
                # Run migrations for local SQLite too
                self._run_migrations(conn)

    def _run_migrations(self, conn) -> None:
        """Add new columns to existing tables if they don't exist."""
        # Migration: Add threat_level column to accounts table
        migrations = [
            # (table, column, column_definition)
            ("accounts", "threat_level", "TEXT DEFAULT 'unknown'"),
            ("accounts", "embedding", "BLOB"),
            ("tweets", "embedding", "BLOB"),
        ]

        for table, column, definition in migrations:
            try:
                # Check if column exists by trying to select it
                conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
            except Exception:
                # Column doesn't exist, add it
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                    print(f"Migration: Added {column} column to {table} table")
                except Exception as e:
                    # May fail if column already exists in some edge case
                    print(f"Migration note: Could not add {column} to {table}: {e}")

        try:
            conn.commit()
        except Exception:
            pass  # May already be committed

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Context manager for database connections."""
        if self.use_turso:
            # Turso connection - libsql has different API
            conn = libsql.connect(
                database=self.turso_url,
                auth_token=self.turso_token
            )
            try:
                yield TursoConnectionWrapper(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            # Local SQLite connection
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()


    # Account operations

    def insert_account(self, account: Account) -> int:
        """Insert an account and return its ID."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO accounts (
                    twitter_id, username, display_name, bio, followers_count,
                    following_count, tweet_count, created_at, profile_image_url,
                    is_verified, location, bot_score, regime_score, coordination_score,
                    account_type, political_stance, political_taxonomy, is_seed,
                    seed_category, features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(twitter_id) DO UPDATE SET
                    username = excluded.username,
                    display_name = excluded.display_name,
                    bio = excluded.bio,
                    followers_count = excluded.followers_count,
                    following_count = excluded.following_count,
                    tweet_count = excluded.tweet_count,
                    created_at = excluded.created_at,
                    profile_image_url = excluded.profile_image_url,
                    is_verified = excluded.is_verified,
                    location = excluded.location,
                    last_updated = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    account.twitter_id,
                    account.username,
                    account.display_name,
                    account.bio,
                    account.followers_count,
                    account.following_count,
                    account.tweet_count,
                    account.created_at.isoformat() if account.created_at else None,
                    account.profile_image_url,
                    account.is_verified,
                    account.location,
                    account.bot_score,
                    account.regime_score,
                    account.coordination_score,
                    account.account_type.value,
                    account.political_stance.value,
                    account.political_taxonomy.value,
                    account.is_seed,
                    account.seed_category.value if account.seed_category else None,
                    json.dumps(account.features) if account.features else None,
                ),
            )
            result = cursor.fetchone()
            return result[0] if result else 0

    def get_account(self, account_id: int) -> Optional[Account]:
        """Get account by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
            return self._row_to_account(row) if row else None

    def get_account_by_username(self, username: str) -> Optional[Account]:
        """Get account by username."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE username = ?", (username.lower(),)
            ).fetchone()
            return self._row_to_account(row) if row else None

    def get_account_by_twitter_id(self, twitter_id: str) -> Optional[Account]:
        """Get account by Twitter ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE twitter_id = ?", (twitter_id,)
            ).fetchone()
            return self._row_to_account(row) if row else None

    def get_all_accounts(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        stance: Optional[PoliticalStance] = None,
        min_bot_score: Optional[float] = None,
        max_bot_score: Optional[float] = None,
        is_seed: Optional[bool] = None,
    ) -> list[Account]:
        """Get accounts with optional filtering."""
        query = "SELECT * FROM accounts WHERE 1=1"
        params: list = []

        if stance:
            query += " AND political_stance = ?"
            params.append(stance.value)
        if min_bot_score is not None:
            query += " AND bot_score >= ?"
            params.append(min_bot_score)
        if max_bot_score is not None:
            query += " AND bot_score <= ?"
            params.append(max_bot_score)
        if is_seed is not None:
            query += " AND is_seed = ?"
            params.append(is_seed)

        query += " ORDER BY last_updated DESC"

        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_account(row) for row in rows]

    def update_account_scores(
        self,
        account_id: int,
        bot_score: Optional[float] = None,
        regime_score: Optional[float] = None,
        coordination_score: Optional[float] = None,
    ) -> None:
        """Update analysis scores for an account."""
        updates = []
        params: list = []

        if bot_score is not None:
            updates.append("bot_score = ?")
            params.append(bot_score)
        if regime_score is not None:
            updates.append("regime_score = ?")
            params.append(regime_score)
        if coordination_score is not None:
            updates.append("coordination_score = ?")
            params.append(coordination_score)

        if not updates:
            return

        updates.append("last_updated = CURRENT_TIMESTAMP")
        params.append(account_id)

        with self.connection() as conn:
            conn.execute(
                f"UPDATE accounts SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    def update_account_classification(
        self,
        account_id: int,
        account_type: Optional[AccountType] = None,
        political_stance: Optional[PoliticalStance] = None,
        political_taxonomy: Optional[PoliticalTaxonomy] = None,
        threat_level: Optional[ThreatLevel] = None,
    ) -> None:
        """Update classification for an account."""
        updates = []
        params: list = []

        if account_type is not None:
            updates.append("account_type = ?")
            params.append(account_type.value)
        if political_stance is not None:
            updates.append("political_stance = ?")
            params.append(political_stance.value)
        if political_taxonomy is not None:
            updates.append("political_taxonomy = ?")
            params.append(political_taxonomy.value)
        if threat_level is not None:
            updates.append("threat_level = ?")
            params.append(threat_level.value)

        if not updates:
            return

        updates.append("last_updated = CURRENT_TIMESTAMP")
        params.append(account_id)

        with self.connection() as conn:
            conn.execute(
                f"UPDATE accounts SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    def update_account_features(self, account_id: int, features: dict) -> None:
        """Update features for an account."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE accounts SET features = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(features), account_id),
            )

    def update_account_embedding(self, account_id: int, embedding: list[float]) -> None:
        """Update embedding for an account.

        Args:
            account_id: Account ID to update
            embedding: List of floats (768-dim for ParsBERT)
        """
        import struct
        embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)

        with self.connection() as conn:
            conn.execute(
                "UPDATE accounts SET embedding = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                (embedding_bytes, account_id),
            )

    def update_tweet_embedding(self, tweet_id: int, embedding: list[float]) -> None:
        """Update embedding for a tweet.

        Args:
            tweet_id: Tweet ID to update
            embedding: List of floats (768-dim for ParsBERT)
        """
        import struct
        embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)

        with self.connection() as conn:
            conn.execute(
                "UPDATE tweets SET embedding = ? WHERE id = ?",
                (embedding_bytes, tweet_id),
            )

    def get_accounts_by_threat_level(
        self,
        threat_level: ThreatLevel,
        limit: Optional[int] = None,
    ) -> list[Account]:
        """Get accounts with a specific threat level."""
        query = "SELECT * FROM accounts WHERE threat_level = ?"
        params: list = [threat_level.value]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_account(row) for row in rows]

    def get_threat_level_stats(self) -> dict[str, int]:
        """Get count of accounts for each threat level."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT threat_level, COUNT(*) as count
                FROM accounts
                WHERE political_stance = 'pro_regime'
                GROUP BY threat_level
                """
            ).fetchall()
            return {row["threat_level"]: row["count"] for row in rows}

    def count_accounts(self) -> int:
        """Count total accounts."""
        with self.connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()
            return result[0] if result else 0

    def _row_to_account(self, row: sqlite3.Row) -> Account:
        """Convert database row to Account model."""
        # Handle embedding - stored as BLOB, convert to list if present
        embedding = None
        try:
            embedding_bytes = row["embedding"] if "embedding" in row.keys() else None
            if embedding_bytes:
                import struct
                # Unpack 768 floats (768 * 4 bytes)
                num_floats = len(embedding_bytes) // 4
                embedding = list(struct.unpack(f'{num_floats}f', embedding_bytes))
        except (KeyError, TypeError):
            embedding = None

        # Handle threat_level - may not exist in older databases
        try:
            threat_level_val = row["threat_level"] if "threat_level" in row.keys() else None
            threat_level = ThreatLevel(threat_level_val) if threat_level_val else ThreatLevel.UNKNOWN
        except (KeyError, ValueError):
            threat_level = ThreatLevel.UNKNOWN

        return Account(
            id=row["id"],
            twitter_id=row["twitter_id"],
            username=row["username"],
            display_name=row["display_name"] or "",
            bio=row["bio"] or "",
            followers_count=row["followers_count"] or 0,
            following_count=row["following_count"] or 0,
            tweet_count=row["tweet_count"] or 0,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            profile_image_url=row["profile_image_url"] or "",
            is_verified=bool(row["is_verified"]),
            location=row["location"] or "",
            bot_score=row["bot_score"],
            regime_score=row["regime_score"],
            coordination_score=row["coordination_score"],
            account_type=AccountType(row["account_type"]) if row["account_type"] else AccountType.UNKNOWN,
            political_stance=PoliticalStance(row["political_stance"]) if row["political_stance"] else PoliticalStance.UNKNOWN,
            political_taxonomy=PoliticalTaxonomy(row["political_taxonomy"]) if row["political_taxonomy"] else PoliticalTaxonomy.UNKNOWN,
            threat_level=threat_level,
            is_seed=bool(row["is_seed"]),
            seed_category=SeedCategory(row["seed_category"]) if row["seed_category"] else None,
            first_seen=datetime.fromisoformat(row["first_seen"]) if row["first_seen"] else None,
            last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None,
            features=json.loads(row["features"]) if row["features"] else None,
            embedding=embedding,
        )

    # Tweet operations

    def insert_tweet(self, tweet: Tweet) -> int:
        """Insert a tweet and return its ID."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tweets (
                    tweet_id, account_id, text, created_at, hashtags, mentions,
                    urls, retweet_count, like_count, reply_count, is_retweet,
                    is_reply, is_quote, referenced_tweet_id, sentiment,
                    regime_alignment, talking_points
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tweet_id) DO UPDATE SET
                    retweet_count = excluded.retweet_count,
                    like_count = excluded.like_count,
                    reply_count = excluded.reply_count
                RETURNING id
                """,
                (
                    tweet.tweet_id,
                    tweet.account_id,
                    tweet.text,
                    tweet.created_at.isoformat() if tweet.created_at else None,
                    json.dumps(tweet.hashtags),
                    json.dumps(tweet.mentions),
                    json.dumps(tweet.urls),
                    tweet.retweet_count,
                    tweet.like_count,
                    tweet.reply_count,
                    tweet.is_retweet,
                    tweet.is_reply,
                    tweet.is_quote,
                    tweet.referenced_tweet_id,
                    tweet.sentiment,
                    tweet.regime_alignment,
                    json.dumps(tweet.talking_points),
                ),
            )
            result = cursor.fetchone()
            return result[0] if result else 0

    def insert_tweets_bulk(self, tweets: list[Tweet]) -> int:
        """Bulk insert tweets. Returns count of inserted tweets."""
        if not tweets:
            return 0

        with self.connection() as conn:
            data = [
                (
                    t.tweet_id,
                    t.account_id,
                    t.text,
                    t.created_at.isoformat() if t.created_at else None,
                    json.dumps(t.hashtags),
                    json.dumps(t.mentions),
                    json.dumps(t.urls),
                    t.retweet_count,
                    t.like_count,
                    t.reply_count,
                    t.is_retweet,
                    t.is_reply,
                    t.is_quote,
                    t.referenced_tweet_id,
                    t.sentiment,
                    t.regime_alignment,
                    json.dumps(t.talking_points),
                )
                for t in tweets
            ]
            conn.executemany(
                """
                INSERT OR IGNORE INTO tweets (
                    tweet_id, account_id, text, created_at, hashtags, mentions,
                    urls, retweet_count, like_count, reply_count, is_retweet,
                    is_reply, is_quote, referenced_tweet_id, sentiment,
                    regime_alignment, talking_points
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )
            return len(data)

    def get_tweet(self, tweet_id: int) -> Optional[Tweet]:
        """Get tweet by ID."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM tweets WHERE id = ?", (tweet_id,)).fetchone()
            return self._row_to_tweet(row) if row else None

    def get_tweets_by_account(
        self, account_id: int, limit: Optional[int] = None
    ) -> list[Tweet]:
        """Get tweets for an account."""
        query = "SELECT * FROM tweets WHERE account_id = ? ORDER BY created_at DESC"
        params: list = [account_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_tweet(row) for row in rows]

    def get_all_tweets(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        since: Optional[datetime] = None,
    ) -> list[Tweet]:
        """Get all tweets with optional filtering."""
        query = "SELECT * FROM tweets WHERE 1=1"
        params: list = []

        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())

        query += " ORDER BY created_at DESC"

        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_tweet(row) for row in rows]

    def search_tweets(self, query: str, limit: int = 100) -> list[Tweet]:
        """Full-text search on tweets."""
        with self.connection() as conn:
            if self.use_turso:
                # Turso: Use LIKE instead of FTS5
                rows = conn.execute(
                    """
                    SELECT * FROM tweets
                    WHERE text LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", limit),
                ).fetchall()
            else:
                # Local SQLite: Use FTS5
                rows = conn.execute(
                    """
                    SELECT tweets.* FROM tweets
                    JOIN tweets_fts ON tweets.id = tweets_fts.rowid
                    WHERE tweets_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
            return [self._row_to_tweet(row) for row in rows]

    def count_tweets(self, account_id: Optional[int] = None) -> int:
        """Count tweets, optionally by account."""
        with self.connection() as conn:
            if account_id:
                result = conn.execute(
                    "SELECT COUNT(*) FROM tweets WHERE account_id = ?", (account_id,)
                ).fetchone()
            else:
                result = conn.execute("SELECT COUNT(*) FROM tweets").fetchone()
            return result[0] if result else 0

    def _row_to_tweet(self, row: sqlite3.Row) -> Tweet:
        """Convert database row to Tweet model."""
        # Handle embedding - stored as BLOB, convert to list if present
        embedding = None
        try:
            embedding_bytes = row["embedding"] if "embedding" in row.keys() else None
            if embedding_bytes:
                import struct
                num_floats = len(embedding_bytes) // 4
                embedding = list(struct.unpack(f'{num_floats}f', embedding_bytes))
        except (KeyError, TypeError):
            embedding = None

        return Tweet(
            id=row["id"],
            tweet_id=row["tweet_id"],
            account_id=row["account_id"],
            text=row["text"] or "",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            hashtags=json.loads(row["hashtags"]) if row["hashtags"] else [],
            mentions=json.loads(row["mentions"]) if row["mentions"] else [],
            urls=json.loads(row["urls"]) if row["urls"] else [],
            retweet_count=row["retweet_count"] or 0,
            like_count=row["like_count"] or 0,
            reply_count=row["reply_count"] or 0,
            is_retweet=bool(row["is_retweet"]),
            is_reply=bool(row["is_reply"]),
            is_quote=bool(row["is_quote"]),
            referenced_tweet_id=row["referenced_tweet_id"],
            sentiment=row["sentiment"],
            regime_alignment=row["regime_alignment"],
            talking_points=json.loads(row["talking_points"]) if row["talking_points"] else [],
            embedding=embedding,
            collected_at=datetime.fromisoformat(row["collected_at"]) if row["collected_at"] else None,
        )

    # Coordination cluster operations

    def insert_cluster(self, cluster: CoordinationCluster) -> int:
        """Insert a coordination cluster."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO coordination_clusters (
                    cluster_type, member_account_ids, evidence,
                    confidence_score, description, is_active
                ) VALUES (?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    cluster.cluster_type.value,
                    json.dumps(cluster.member_account_ids),
                    json.dumps(cluster.evidence),
                    cluster.confidence_score,
                    cluster.description,
                    cluster.is_active,
                ),
            )
            result = cursor.fetchone()
            return result[0] if result else 0

    def get_clusters(
        self,
        cluster_type: Optional[CoordinationType] = None,
        min_confidence: float = 0.0,
        is_active: Optional[bool] = None,
    ) -> list[CoordinationCluster]:
        """Get coordination clusters."""
        query = "SELECT * FROM coordination_clusters WHERE confidence_score >= ?"
        params: list = [min_confidence]

        if cluster_type:
            query += " AND cluster_type = ?"
            params.append(cluster_type.value)
        if is_active is not None:
            query += " AND is_active = ?"
            params.append(is_active)

        query += " ORDER BY confidence_score DESC"

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_cluster(row) for row in rows]

    def count_clusters(self) -> int:
        """Count total coordination clusters."""
        with self.connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM coordination_clusters").fetchone()
            return result[0] if result else 0

    def _row_to_cluster(self, row: sqlite3.Row) -> CoordinationCluster:
        """Convert database row to CoordinationCluster model."""
        return CoordinationCluster(
            id=row["id"],
            cluster_type=CoordinationType(row["cluster_type"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            member_account_ids=json.loads(row["member_account_ids"]) if row["member_account_ids"] else [],
            evidence=json.loads(row["evidence"]) if row["evidence"] else {},
            confidence_score=row["confidence_score"] or 0.0,
            description=row["description"] or "",
            is_active=bool(row["is_active"]),
        )

    # Seed account operations

    def insert_seed(self, seed: SeedAccount) -> int:
        """Insert a seed account."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO seed_accounts (username, category, notes)
                VALUES (?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    category = excluded.category,
                    notes = excluded.notes
                RETURNING id
                """,
                (seed.username.lower(), seed.category.value, seed.notes),
            )
            result = cursor.fetchone()
            return result[0] if result else 0

    def get_seeds(self, category: Optional[SeedCategory] = None) -> list[SeedAccount]:
        """Get seed accounts."""
        query = "SELECT * FROM seed_accounts"
        params: list = []

        if category:
            query += " WHERE category = ?"
            params.append(category.value)

        query += " ORDER BY added_at"

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                SeedAccount(
                    id=row["id"],
                    username=row["username"],
                    category=SeedCategory(row["category"]),
                    added_at=datetime.fromisoformat(row["added_at"]) if row["added_at"] else None,
                    notes=row["notes"] or "",
                )
                for row in rows
            ]

    def remove_seed(self, username: str) -> bool:
        """Remove a seed account."""
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM seed_accounts WHERE username = ?", (username.lower(),)
            )
            return cursor.rowcount > 0

    def count_seeds(self) -> int:
        """Count seed accounts."""
        with self.connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM seed_accounts").fetchone()
            return result[0] if result else 0

    # Utility methods

    def get_stats(self) -> dict:
        """Get database statistics."""
        return {
            "accounts": self.count_accounts(),
            "tweets": self.count_tweets(),
            "clusters": self.count_clusters(),
            "seeds": self.count_seeds(),
        }


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get or create global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db
