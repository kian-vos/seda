# SEDA User Manual

**State-linked Entity Detection & Analysis**

A platform for identifying Iranian regime-affiliated Twitter accounts through bot detection, stance classification, and coordination analysis.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Quick Start](#quick-start)
4. [Data Collection](#data-collection)
5. [Network Expansion](#network-expansion)
6. [Analysis Pipeline](#analysis-pipeline)
7. [Dashboard](#dashboard)
8. [Workflow Examples](#workflow-examples)
9. [Command Reference](#command-reference)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.11 or higher
- Apify account (for Twitter scraping)
- Anthropic API key (optional, for LLM-based stance classification)

### Setup

```bash
# Clone or navigate to the project
cd seda

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Apify API token for Twitter scraping
APIFY_API_TOKEN=your_apify_token_here

# Optional: Anthropic API key for LLM stance classification
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Database path (default: data/seda.db)
DATABASE_PATH=data/seda.db
```

### Getting API Keys

**Apify Token:**
1. Create account at https://apify.com
2. Go to Settings → Integrations → API tokens
3. Create and copy your token
4. Recommended: Upgrade to paid plan for higher limits

**Anthropic Key:**
1. Create account at https://console.anthropic.com
2. Navigate to API Keys
3. Create and copy your key

### Apify Actors Used

SEDA uses two premium Apify actors for comprehensive Twitter scraping:

| Actor | Purpose | Pricing |
|-------|---------|---------|
| `apidojo/tweet-scraper` | Tweets, profiles, search | ~$1.50/1000 tweets |
| `apidojo/twitter-user-scraper` | Followers, following, retweeters | ~$0.30/1000 users |

These actors provide:
- **Retweeters** - Find who amplifies regime content
- **Followers** - Find who follows regime accounts
- **Following** - Find regime account networks
- **Search** - Find tweets by hashtag/keyword
- **Profiles & Tweets** - Collect account data

---

## Quick Start

Run these commands in order for a complete analysis:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Initialize seed accounts (known regime-affiliated accounts)
python -m scripts.seed_accounts init

# 3. Collect data from seed accounts
python -m scripts.collect seeds --tweets 50

# 4. Expand network to find amplifiers
python -m scripts.collect followers --all-seeds --max 200
python -m scripts.collect expand --depth 1 --retweeters 100

# 5. Run full analysis pipeline
python -m scripts.analyze all --no-llm

# 6. View results in dashboard
streamlit run dashboard/app.py
```

---

## Data Collection

### Seed Account Management

Seed accounts are known regime-affiliated accounts used as starting points.

```bash
# Initialize default seed accounts (49 verified regime accounts)
python -m scripts.seed_accounts init

# List all seed accounts
python -m scripts.seed_accounts list

# Add a new seed account
python -m scripts.seed_accounts add USERNAME CATEGORY
# Categories: official_state, state_media, irgc_linked, other

# Remove a seed account
python -m scripts.seed_accounts remove USERNAME
```

### Collecting Twitter Data

```bash
# Scrape a single account and their tweets
python -m scripts.collect account USERNAME --tweets 100

# Scrape all seed accounts
python -m scripts.collect seeds --tweets 50

# Scrape seeds by category
python -m scripts.collect seeds --category state_media --tweets 50

# Scrape accounts from a file (one username per line)
python -m scripts.collect batch usernames.txt --tweets 100

# Search tweets by query (also saves accounts found)
python -m scripts.collect search "query here" --max 100

# View collection statistics
python -m scripts.collect stats
```

---

## Network Expansion

The key to finding thousands of regime-affiliated accounts is **network expansion** - finding accounts that interact with known regime accounts.

### Strategy 1: Followers (Recommended First)

Find accounts that follow regime accounts - these are potential amplifiers.

```bash
# Get followers of a specific account
python -m scripts.collect followers PressTV --max 500

# Get followers from ALL seed accounts
python -m scripts.collect followers --all-seeds --max 200

# Expected yield: 100-500 new accounts per seed
```

### Strategy 2: Retweeters

Find accounts that retweet regime content - these are active amplifiers.

```bash
# Find retweeters of seed account tweets
python -m scripts.collect expand --depth 1 --tweets 10 --retweeters 100

# Go deeper (retweeters of retweeters)
python -m scripts.collect expand --depth 2 --tweets 5 --retweeters 50

# Options:
#   --depth N      : Levels of expansion (default: 1)
#   --tweets N     : Tweets per seed to check (default: 10)
#   --retweeters N : Max retweeters per tweet (default: 50)
```

### Strategy 3: Hashtag/Keyword Search

Find accounts using pro-regime hashtags and keywords.

```bash
# Search for pro-regime Persian hashtags
python -m scripts.collect search "#محور_مقاومت" --max 500
python -m scripts.collect search "#استکبار" --max 500
python -m scripts.collect search "رهبر معظم" --max 500

# Search for English regime content
python -m scripts.collect search "axis of resistance" --max 500
python -m scripts.collect search "from:PressTV" --max 200
```

### Strategy 4: Batch Collection

If you have a list of suspected accounts:

```bash
# Create a file with usernames (one per line)
echo "suspect1
suspect2
suspect3" > suspects.txt

# Collect all accounts
python -m scripts.collect batch suspects.txt --tweets 50
```

### Scaling to Thousands of Accounts

For large-scale collection:

```bash
# Step 1: Collect seeds (30-50 accounts)
python -m scripts.collect seeds --tweets 20

# Step 2: Get followers of all seeds (1000-5000 accounts)
python -m scripts.collect followers --all-seeds --max 500

# Step 3: Find retweeters (1000-3000 more accounts)
python -m scripts.collect expand --depth 2 --tweets 10 --retweeters 200

# Step 4: Search hashtags (500-1000 more accounts)
python -m scripts.collect search "#IRGC" --max 500
python -m scripts.collect search "#محور_مقاومت" --max 500

# Step 5: Run analysis
python -m scripts.analyze all --no-llm
```

---

## Analysis Pipeline

### Individual Analysis Commands

```bash
# Extract behavioral features for bot detection
python -m scripts.analyze features --limit 100

# Run bot detection scoring
python -m scripts.analyze bot --limit 100

# Train bot detection model (requires labeled data)
python -m scripts.analyze bot --train

# Use heuristic bot scoring (no ML model needed)
python -m scripts.analyze bot --heuristic

# Run stance classification (rule-based)
python -m scripts.analyze stance --limit 100

# Run stance classification with LLM (costs API credits)
python -m scripts.analyze stance --llm --limit 100

# Detect coordination patterns
python -m scripts.analyze coordination --window 300 --similarity 0.7
```

### Full Pipeline

```bash
# Run all analysis steps
python -m scripts.analyze all

# Run without LLM (rule-based stance classification only)
python -m scripts.analyze all --no-llm

# Limit number of accounts processed
python -m scripts.analyze all --limit 100

# Force recomputation of existing results
python -m scripts.analyze all --force
```

### Reports and Export

```bash
# Generate analysis report
python -m scripts.analyze report

# Explain scores for a specific account
python -m scripts.analyze explain USERNAME

# Export results to CSV
python -m scripts.analyze export results.csv

# Export to JSON
python -m scripts.analyze export results.json

# Export with filters
python -m scripts.analyze export results.csv --min-bot 0.5 --stance pro_regime --limit 1000
```

---

## Dashboard

Launch the Streamlit dashboard for interactive exploration:

```bash
streamlit run dashboard/app.py
```

The dashboard opens at http://localhost:8501 and includes:

### Pages

| Page | Description |
|------|-------------|
| **Overview** | Summary statistics, stance distribution, pro-regime account list |
| **Accounts** | Searchable table with filters, click to see details |
| **Coordination** | Browse detected coordination clusters |
| **Network** | Treemap visualization of seed accounts |
| **Collect More** | Guide for scaling up data collection |
| **Export** | Download filtered data as CSV/JSON |

### Filters

- Bot score range (0.0 - 1.0)
- Political stance (pro_regime, anti_regime, neutral)
- Account type (official_state, state_media, irgc_linked, amplifier, unknown)
- Seed accounts only
- Text search (username, bio, tweets)

---

## Workflow Examples

### Example 1: Basic Analysis of Known Accounts

```bash
# Setup
source .venv/bin/activate
python -m scripts.seed_accounts init

# Collect seed data
python -m scripts.collect seeds --tweets 100

# Analyze
python -m scripts.analyze all --no-llm

# View report
python -m scripts.analyze report
```

### Example 2: Find Hidden Amplifier Networks (Recommended)

```bash
# Start with seeds
python -m scripts.collect seeds --tweets 20

# Get followers of regime accounts
python -m scripts.collect followers --all-seeds --max 300

# Find retweeters of regime content
python -m scripts.collect expand --depth 1 --retweeters 100

# Run full analysis
python -m scripts.analyze all --no-llm

# Check for coordination clusters
python -m scripts.analyze coordination

# View in dashboard
streamlit run dashboard/app.py
```

### Example 3: Large-Scale Collection (1000+ accounts)

```bash
# Collect seeds
python -m scripts.collect seeds --tweets 20

# Mass follower collection
python -m scripts.collect followers --all-seeds --max 500

# Retweeter expansion
python -m scripts.collect expand --depth 2 --tweets 10 --retweeters 200

# Hashtag searches
python -m scripts.collect search "#محور_مقاومت" --max 500
python -m scripts.collect search "axis of resistance" --max 500

# Full analysis
python -m scripts.analyze all --no-llm

# Generate report
python -m scripts.analyze report
```

### Example 4: Investigate a Specific Account

```bash
# Collect account data
python -m scripts.collect account suspicious_user --tweets 200

# Run analysis
python -m scripts.analyze features
python -m scripts.analyze bot
python -m scripts.analyze stance

# Get detailed explanation
python -m scripts.analyze explain suspicious_user
```

### Example 5: LLM-Enhanced Stance Classification

```bash
# Ensure ANTHROPIC_API_KEY is set in .env

# Run stance classification with Claude
python -m scripts.analyze stance --llm --limit 50

# Or run full pipeline with LLM
python -m scripts.analyze all  # (without --no-llm flag)
```

---

## Command Reference

### scripts.seed_accounts

| Command | Description |
|---------|-------------|
| `init` | Load default Iranian regime seed accounts |
| `list` | Show all configured seed accounts |
| `add USERNAME CATEGORY` | Add a seed account |
| `remove USERNAME` | Remove a seed account |

### scripts.collect

| Command | Description |
|---------|-------------|
| `account USERNAME` | Scrape single account + tweets |
| `seeds` | Scrape all seed accounts |
| `followers USERNAME` | Get followers of an account |
| `followers --all-seeds` | Get followers of all seed accounts |
| `expand` | Find retweeters to expand network |
| `batch FILE` | Scrape accounts from file |
| `search QUERY` | Search tweets by query |
| `stats` | Show collection statistics |

### scripts.analyze

| Command | Description |
|---------|-------------|
| `features` | Extract behavioral features |
| `bot` | Run bot detection |
| `stance` | Run stance classification |
| `coordination` | Detect coordination patterns |
| `all` | Run full analysis pipeline |
| `report` | Generate summary report |
| `explain USERNAME` | Explain account scores |
| `export FILE` | Export results to CSV/JSON |

---

## Understanding the Scores

### Bot Score (0.0 - 1.0)

Probability that an account exhibits bot-like behavior.

| Score | Interpretation |
|-------|----------------|
| 0.0 - 0.3 | Likely human |
| 0.3 - 0.5 | Uncertain |
| 0.5 - 0.7 | Suspicious |
| 0.7 - 1.0 | Likely bot |

**Features analyzed:**
- Posting patterns (timing, frequency, regularity)
- Profile characteristics (age, bio, avatar)
- Content behavior (retweet ratio, hashtag usage)
- Engagement patterns (likes, replies received)

### Political Stance

| Stance | Description |
|--------|-------------|
| `pro_regime` | Supports Iranian government/IRGC |
| `anti_regime` | Opposes Iranian government |
| `neutral` | No clear political alignment |
| `unknown` | Insufficient data to classify |

### Political Taxonomy (detailed classification)

| Taxonomy | Description |
|----------|-------------|
| `principlist` | Conservative pro-regime |
| `reformist` | Reform-oriented pro-regime |
| `irgc_aligned` | IRGC-affiliated |
| `monarchist` | Pro-monarchy opposition |
| `republican` | Democratic opposition |
| `mek` | MEK-affiliated |
| `leftist` | Left-wing opposition |
| `diaspora` | Iranian diaspora |
| `unknown` | Unclassified |

### Coordination Score (0.0 - 1.0)

Likelihood of participating in coordinated inauthentic behavior.

**Detection methods:**
- Synchronized posting (similar content within time windows)
- Hashtag campaign participation
- Amplification network membership

### Account Types

| Type | Description |
|------|-------------|
| `official_state` | Government officials, ministries |
| `state_media` | State-owned media outlets |
| `irgc_linked` | IRGC-affiliated accounts |
| `amplifier` | Accounts that amplify regime content |
| `unknown` | Unclassified accounts |

---

## Troubleshooting

### "APIFY_API_TOKEN not set"

Ensure your `.env` file exists and contains:
```
APIFY_API_TOKEN=your_token_here
```

### "ModuleNotFoundError: No module named 'seda'"

Install the package in development mode:
```bash
pip install -e .
```

### "Failed to scrape @username"

- Account may be suspended, private, or renamed
- Check Apify quota in your dashboard
- Verify the username is correct (without @ symbol)

### No coordination clusters detected

Coordination detection requires:
- Multiple accounts (ideally 100+)
- Amplifier accounts (run `followers` and `expand` commands)
- Similar posting behavior within time windows

### Dashboard won't start

```bash
# Ensure streamlit is installed
pip install streamlit

# Run from project root
streamlit run dashboard/app.py
```

### Slow feature extraction

Feature extraction processes all tweets per account. Use `--limit` to process fewer accounts:
```bash
python -m scripts.analyze features --limit 50
```

### Apify actor errors

The actors used are:
- `apidojo/tweet-scraper` - For tweets and search
- `apidojo/twitter-user-scraper` - For followers and retweeters

Check the Apify console for run logs if scraping fails.

---

## Data Storage

All data is stored in SQLite at `data/seda.db`:

| Table | Contents |
|-------|----------|
| `accounts` | Twitter accounts with scores |
| `tweets` | Collected tweets |
| `seed_accounts` | Configured seed accounts |
| `coordination_clusters` | Detected coordination groups |
| `tweets_fts` | Full-text search index |

To reset the database:
```bash
rm data/seda.db
python -m scripts.seed_accounts init
```

---

## Cost Estimates

### Apify Costs (Pay-per-result)

| Actor | Cost | Example |
|-------|------|---------|
| tweet-scraper | ~$1.50/1000 tweets | 10K tweets = ~$15 |
| twitter-user-scraper | ~$0.30/1000 users | 5K followers = ~$1.50 |

### Claude API Costs

| Usage | Cost |
|-------|------|
| 1000 stance classifications | ~$2-5 |

### Budget Examples

| Scale | Accounts | Est. Cost |
|-------|----------|-----------|
| Small | 500 | ~$5-10 |
| Medium | 2,000 | ~$15-25 |
| Large | 10,000 | ~$50-100 |

Rule-based stance classification (`--no-llm`) has no API cost.

---

## Technical Details

### Apify Actors

SEDA uses these Apify actors:

**apidojo/tweet-scraper**
- Scrapes user timelines and search results
- Returns tweets with full metadata
- Supports search by keyword/hashtag

**apidojo/twitter-user-scraper**
- Scrapes followers and following lists
- Scrapes retweeters of specific tweets
- Pay-per-result pricing ($0.30/1000 users)

### Data Flow

```
Seeds → Followers/Retweeters → Analysis → Coordination Detection
  ↓           ↓                    ↓              ↓
Tweets    Amplifiers          Bot Scores     Clusters
```

---

## Support

- Database: `data/seda.db`
- Logs: Check terminal output for errors

For issues, check:
1. API tokens are correctly set in `.env`
2. Virtual environment is activated
3. Package is installed (`pip install -e .`)
4. Apify account has sufficient credits
