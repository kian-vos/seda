# SEDA Technical Documentation

This document covers the technical implementation details of SEDA. For an overview of what SEDA does and why, see the [README](README.md).

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Detection Methods](#detection-methods)
- [CLI Commands](#cli-commands)
- [Database Schema](#database-schema)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Development](#development)

---

## Architecture

```
seda/
├── seda/                       # Core package
│   ├── analysis/               # Detection algorithms
│   │   ├── bot.py              # Bot detection (LightGBM + heuristics)
│   │   ├── stance.py           # Political stance classification
│   │   ├── coordination.py     # Coordination detection (MinHash, Louvain)
│   │   ├── nlp.py              # Persian NLP and keyword detection
│   │   ├── features.py         # Feature extraction (37 features)
│   │   └── embeddings.py       # ParsBERT semantic embeddings
│   ├── scraper/
│   │   └── twitter.py          # Apify Twitter scraper
│   ├── db.py                   # SQLite/Turso database
│   ├── models.py               # Pydantic data models
│   └── config.py               # Settings management
│
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   └── network_viz.py          # Network graph visualizations
│
├── scripts/
│   ├── collect.py              # Data collection CLI
│   ├── analyze.py              # Analysis pipeline CLI
│   └── seed_accounts.py        # Seed account management
│
└── data/
    ├── seda.db                 # SQLite database
    └── models/                 # Trained ML models
```

---

## Installation

### Prerequisites

- Python 3.11+
- [Apify](https://apify.com) account (free tier: 10K results/month)
- [Anthropic API key](https://console.anthropic.com) (optional, for LLM classification)

### Basic Installation

```bash
git clone https://github.com/kian-vos/seda.git
cd seda
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Full Installation (with ML features)

```bash
pip install -e ".[ml]"
# Or manually:
pip install lightgbm scikit-learn networkx transformers torch shap
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
APIFY_API_TOKEN=your_apify_token
ANTHROPIC_API_KEY=your_anthropic_key  # Optional
```

---

## Detection Methods

### 1. Bot Detection

SEDA uses a hybrid approach: LightGBM classifier (when trained) with heuristic fallback.

#### Features (37 total)

| Category | Features |
|----------|----------|
| **Profile** | username_digit_ratio, username_length, bio_length, has_bio, has_default_avatar, account_age_days, follower_following_ratio, tweet_rate, is_verified |
| **Timing** | hour_entropy, night_posting_ratio, post_interval_variance, periodicity_score |
| **Content** | retweet_ratio, duplicate_content_ratio, url_ratio, hashtag_density, mention_ratio |
| **Engagement** | zero_engagement_ratio, engagement_rate, reply_ratio |

#### Training with Weak Supervision

```bash
# Train using automatically generated labels
python -m scripts.analyze train-bot --weak --cv 5
```

The weak supervision labeler uses high-confidence signals:
- **Likely bots**: >95% retweet ratio, new account + high volume, in coordination cluster
- **Likely humans**: Verified, old account, varied content, good engagement

### 2. Stance Classification

Accounts are classified as `pro_regime`, `anti_regime`, `neutral`, or `unknown`.

#### Keywords

**Pro-Regime (Persian):**
- رهبر معظم (Supreme Leader)
- محور مقاومت (Axis of Resistance)
- سپاه (IRGC)
- بسیج (Basij)
- استکبار (Arrogance/imperialism)

**Opposition (Persian):**
- زن زندگی آزادی (Woman Life Freedom)
- مرگ بر دیکتاتور (Death to Dictator)
- مهسا امینی (Mahsa Amini)
- سرنگونی (Overthrow)

#### LLM Classification

For ambiguous accounts, SEDA can use Claude for nuanced classification:

```bash
python -m scripts.analyze stance --llm
```

### 3. Threat Level Classification

Pro-regime accounts are further categorized by threat level:

| Level | Detection Signals |
|-------|-------------------|
| `violence_inciter` | Keywords: اعدام کنید, بکشید, تیرباران (execute, kill, firing squad) |
| `doxxer` | Keywords: شناسایی شد, هویت, آدرس (identified, identity, address) |
| `irgc_operative` | IRGC keywords + commander names + proxy groups |
| `state_propagandist` | Seed accounts in official_state/state_media categories |
| `amplifier_bot` | Bot score >= 0.7 |
| `troll` | Harassment keywords >= 3 |
| `passive_supporter` | No violence/harassment signals |

### 4. Coordination Detection

#### Synchronized Posting (MinHash LSH)

Detects near-duplicate tweets posted within a time window.

```python
# Configuration
minhash_threshold = 0.7      # Similarity threshold
minhash_num_perm = 128       # MinHash permutations
coordination_time_window = 5  # Minutes
```

#### Hashtag Campaign Detection

Identifies velocity spikes in hashtag usage driven by small clusters.

#### Amplification Networks (Louvain)

Graph-based community detection on retweet/mention networks.

### 5. Semantic Embeddings (ParsBERT)

Optional 768-dimensional embeddings using `HooshvareLab/bert-fa-base-uncased`.

```bash
# Generate embeddings
python -m scripts.analyze embed --limit 100

# Find similar accounts
python -m scripts.analyze similarity @PressTV --top 10
```

**Use cases:**
- Semantic similarity search
- Content clustering
- Finding accounts with similar messaging

---

## CLI Commands

### Data Collection (`scripts/collect.py`)

```bash
# Collect single account
python -m scripts.collect account @PressTV --tweets 100

# Collect all seed accounts
python -m scripts.collect seeds --tweets 50

# Collect followers
python -m scripts.collect followers @PressTV --max 500
python -m scripts.collect followers --all-seeds --max 200

# Network expansion (retweeters)
python -m scripts.collect expand --depth 2 --tweets 20 --retweeters 200

# Search tweets
python -m scripts.collect search "#محور_مقاومت" --max 500
```

### Analysis (`scripts/analyze.py`)

```bash
# Run full pipeline
python -m scripts.analyze all --no-llm

# Individual steps
python -m scripts.analyze features    # Extract features
python -m scripts.analyze bot         # Bot detection
python -m scripts.analyze stance      # Stance classification
python -m scripts.analyze coordination # Coordination detection

# ML training
python -m scripts.analyze train-bot --weak --cv 5

# Reports
python -m scripts.analyze report      # Summary report
python -m scripts.analyze threat      # Threat level breakdown
python -m scripts.analyze explain @username  # Explain scores

# Embeddings
python -m scripts.analyze embed --limit 100 --tweets
python -m scripts.analyze similarity @username --top 10

# Export
python -m scripts.analyze export results.csv --stance pro_regime
```

### Seed Management (`scripts/seed_accounts.py`)

```bash
python -m scripts.seed_accounts init      # Initialize default seeds
python -m scripts.seed_accounts list      # List all seeds
python -m scripts.seed_accounts add @username --category state_media
python -m scripts.seed_accounts remove @username
```

---

## Database Schema

### Accounts Table

```sql
CREATE TABLE accounts (
    id INTEGER PRIMARY KEY,
    twitter_id TEXT UNIQUE NOT NULL,
    username TEXT NOT NULL,
    display_name TEXT,
    bio TEXT,
    followers_count INTEGER,
    following_count INTEGER,
    tweet_count INTEGER,
    created_at TIMESTAMP,
    profile_image_url TEXT,
    is_verified BOOLEAN,
    location TEXT,

    -- Analysis scores
    bot_score REAL,
    regime_score REAL,
    coordination_score REAL,

    -- Classifications
    account_type TEXT,        -- unknown, official_state, state_media, irgc_linked, amplifier, bot
    political_stance TEXT,    -- pro_regime, anti_regime, neutral, unknown
    political_taxonomy TEXT,  -- principlist, reformist, irgc_aligned, monarchist, etc.
    threat_level TEXT,        -- violence_inciter, doxxer, irgc_operative, etc.

    -- Seed tracking
    is_seed BOOLEAN,
    seed_category TEXT,

    -- Storage
    features TEXT,            -- JSON blob of extracted features
    embedding BLOB,           -- 768 floats (ParsBERT)

    first_seen TIMESTAMP,
    last_updated TIMESTAMP
);
```

### Tweets Table

```sql
CREATE TABLE tweets (
    id INTEGER PRIMARY KEY,
    tweet_id TEXT UNIQUE NOT NULL,
    account_id INTEGER REFERENCES accounts(id),
    text TEXT,
    created_at TIMESTAMP,
    hashtags TEXT,            -- JSON array
    mentions TEXT,            -- JSON array
    urls TEXT,                -- JSON array
    retweet_count INTEGER,
    like_count INTEGER,
    reply_count INTEGER,
    is_retweet BOOLEAN,
    is_reply BOOLEAN,
    is_quote BOOLEAN,
    referenced_tweet_id TEXT,

    -- Analysis
    sentiment REAL,
    regime_alignment REAL,
    talking_points TEXT,      -- JSON array
    embedding BLOB,           -- 768 floats (ParsBERT)

    collected_at TIMESTAMP
);
```

---

## API Reference

### Analysis Classes

```python
from seda.analysis import (
    BotDetector,
    StanceClassifier,
    CoordinationDetector,
    FeatureExtractor,
    PersianNLP,
)

# Bot detection
detector = BotDetector()
score = detector.score_account(account)
explanation = detector.explain_prediction(account)

# Stance classification
classifier = StanceClassifier()
stance, taxonomy, threat_level = classifier.classify_account(account, tweets)

# Feature extraction
extractor = FeatureExtractor()
features = extractor.extract_account_features(account, tweets)

# NLP
nlp = PersianNLP()
signals = nlp.get_threat_signals(text)
keywords = nlp.detect_violence_keywords(text)
```

### Embeddings

```python
from seda.analysis.embeddings import PersianEmbedder

embedder = PersianEmbedder()
embedding = embedder.embed_text("some Persian text")
similar = embedder.find_similar_accounts(account, top_k=10)
```

### Database

```python
from seda.db import get_db

db = get_db()
accounts = db.get_all_accounts(stance=PoliticalStance.PRO_REGIME, limit=100)
tweets = db.get_tweets_by_account(account_id, limit=50)
stats = db.get_threat_level_stats()
```

---

## Deployment

### Local Development

Data stored in `data/seda.db` (SQLite). Works great for development and research.

### Streamlit Cloud

**Warning:** Streamlit Cloud has ephemeral storage. Database resets on redeploy.

Options:
1. Pre-load from CSV on startup
2. Use external database (Turso, Supabase, PlanetScale)

### Turso (Edge SQLite)

```env
TURSO_DATABASE_URL=libsql://your-db.turso.io
TURSO_AUTH_TOKEN=your_token
```

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
ruff check .
ruff format .
```

### Adding New Keywords

Edit `seda/analysis/nlp.py`:

```python
# Add to appropriate list
PRO_REGIME_KEYWORDS_FA = [
    "existing_keyword",
    "new_keyword",  # Add here
]
```

### Adding New Features

1. Add feature extraction in `seda/analysis/features.py`
2. Update `FEATURE_NAMES` list in `seda/analysis/bot.py`
3. Retrain model: `python -m scripts.analyze train-bot`

---

## Resource Requirements

| Component | CPU | GPU | Memory | Storage |
|-----------|-----|-----|--------|---------|
| Basic analysis | Fast | N/A | 1GB | 100MB |
| LightGBM training | Fast | N/A | 2GB | 10MB |
| ParsBERT embeddings | Slow | Fast | 4GB | 500MB |
| Full dashboard | Fast | N/A | 2GB | Varies |

**Minimum:** MacBook with 8GB RAM
**Recommended:** 16GB RAM, GPU for embeddings

---

## Troubleshooting

### Database schema errors

```bash
# Add new columns to existing database
sqlite3 data/seda.db "ALTER TABLE accounts ADD COLUMN threat_level TEXT DEFAULT 'unknown';"
sqlite3 data/seda.db "ALTER TABLE accounts ADD COLUMN embedding BLOB;"
```

### Import errors

```bash
# Make sure you're in the seda directory
cd /path/to/seda
pip install -e .
```

### Streamlit import issues

```bash
# Run from project root
cd /path/to/seda
streamlit run dashboard/app.py
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.
