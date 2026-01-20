<p align="center">
  <img src="dashboard/assets/seda_logo.png" alt="SEDA Logo" width="300">
</p>

<h1 align="center">SEDA</h1>
<h3 align="center">State-linked Entity Detection & Analysis</h3>

<p align="center">
  <strong>Exposing Iranian regime propaganda networks on social media</strong>
</p>

<p align="center">
  <a href="#the-problem">The Problem</a> •
  <a href="#what-seda-does">What SEDA Does</a> •
  <a href="#threat-categories">Threat Categories</a> •
  <a href="#get-started">Get Started</a> •
  <a href="#join-the-fight">Join the Fight</a>
</p>

---

## The Problem

During the Woman, Life, Freedom movement and other moments of Iranian resistance, a hidden army operates in the shadows of social media.

**Thousands of accounts** — some automated, some operated by regime loyalists — work around the clock to:

- **Drown out protesters' voices** with coordinated hashtag campaigns
- **Spread regime propaganda** disguised as organic public opinion
- **Identify and doxx activists**, putting lives at risk
- **Incite violence** against those who speak out
- **Create the illusion of public support** for a regime that murders its own people

These aren't random trolls. They're organized networks — some directly linked to the IRGC (Islamic Revolutionary Guard Corps), others operating through state media, and many hiding behind fake personas.

**They operate with impunity because they're hard to find.**

Until now.

---

## What SEDA Does

SEDA is an open-source intelligence platform that **identifies, categorizes, and exposes** regime-affiliated accounts.

### Unmask the Networks

SEDA analyzes account behavior, content, and connections to identify:

- **Automated bots** that amplify propaganda 24/7
- **IRGC-connected accounts** spreading military narratives
- **State media amplifiers** retweeting official propaganda
- **Violence inciters** calling for harm against protesters
- **Doxxers** who expose activists' identities
- **Coordinated campaigns** that manipulate trending topics

### Understand the Threat

Not all regime accounts are equal. SEDA categorizes threats by severity:

| Threat Level | Description | Risk |
|--------------|-------------|------|
| **Violence Inciters** | Call for execution, violence against protesters | CRITICAL |
| **Doxxers** | Expose opposition identities and personal information | CRITICAL |
| **IRGC Operatives** | Direct connections to Revolutionary Guard | HIGH |
| **State Propagandists** | Official media and government accounts | HIGH |
| **Amplifier Bots** | Automated accounts that spread content | MEDIUM |
| **Trolls** | Harassment and intimidation campaigns | MEDIUM |
| **Passive Supporters** | Engage with regime content without inciting | LOW |

### Visualize Connections

See how regime accounts connect and coordinate:

- **Network graphs** showing who amplifies whom
- **Coordination clusters** revealing synchronized campaigns
- **Threat breakdowns** showing the composition of regime networks

---

## Why This Matters

### For Researchers
- Study information warfare tactics in real-time
- Document regime propaganda for academic research
- Understand how authoritarian states manipulate social media

### For Journalists
- Verify whether trending topics are organic or manufactured
- Identify state-linked accounts before amplifying their content
- Investigate coordination between regime entities

### For Activists
- Know who's watching and who's dangerous
- Understand which accounts incite violence or doxx
- Protect yourself by understanding the threat landscape

### For Platforms
- Identify coordinated inauthentic behavior
- Understand how state actors evade detection
- Improve content moderation with behavioral signals

---

## Get Started

### Quick Setup

```bash
# Clone and install
git clone https://github.com/kian-vos/seda.git
cd seda
pip install -e .

# Configure (get free API keys from Apify)
cp .env.example .env

# Initialize and collect data
python -m scripts.seed_accounts init
python -m scripts.collect seeds --tweets 50

# Run analysis
python -m scripts.analyze all

# Launch dashboard
streamlit run dashboard/app.py
```

### View the Dashboard

Open `http://localhost:8501` to explore:

- **Overview**: See threat breakdown and key statistics
- **Accounts**: Search and filter identified regime accounts
- **Network**: Visualize connections and coordination
- **Export**: Download data for your own analysis

For detailed technical documentation, see [TECHNICAL.md](TECHNICAL.md).

---

## Join the Fight

SEDA is open source because **transparency fights propaganda**.

### Ways to Contribute

**No coding required:**
- **Report accounts**: Know regime-affiliated accounts? Help us add them
- **Improve keywords**: Suggest Persian/English terms that indicate regime affiliation
- **Spread the word**: Share SEDA with researchers and journalists
- **Translate**: Help make SEDA accessible in more languages

**For developers:**
- Improve detection algorithms
- Add new analysis capabilities
- Enhance the dashboard
- Fix bugs and improve performance

See our [Contributing Guide](CONTRIBUTING.md) for details.

---

## The Bigger Picture

SEDA is one tool in a larger fight for information freedom.

Every regime account exposed is:
- One less voice drowning out protesters
- One less source of manufactured consent
- One more data point proving coordinated manipulation

**Information warfare is real. Now we can fight back.**

---

## Ethics & Responsibility

SEDA is designed for **research, journalism, and accountability** — not harassment.

- We focus on **regime-affiliated accounts**, not individuals' political opinions
- We **protect opposition voices** by excluding anti-regime accounts from analysis
- We encourage **responsible disclosure** and ethical use of findings
- Users must comply with platform terms of service and applicable laws

---

## Acknowledgments

Built for the brave people of Iran who risk everything for freedom.

*"Woman, Life, Freedom"*
*"زن، زندگی، آزادی"*

---

<p align="center">
  <strong>For those who fight for freedom.</strong>
</p>

<p align="center">
  <a href="TECHNICAL.md">Technical Documentation</a> •
  <a href="LICENSE">MIT License</a>
</p>
