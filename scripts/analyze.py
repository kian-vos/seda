"""Analysis pipeline CLI."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from seda.config import get_settings

settings = get_settings()
from seda.db import get_db
from seda.analysis import (
    BotDetector,
    CoordinationDetector,
    FeatureExtractor,
    StanceClassifier,
    is_embeddings_available,
)
from seda.models import ThreatLevel, PoliticalStance

app = typer.Typer(help="Run analysis pipelines")
console = Console()


@app.command()
def features(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit accounts to process"),
):
    """Extract features for bot detection."""
    extractor = FeatureExtractor()
    db = get_db()

    # Get account IDs
    if limit:
        accounts = db.get_all_accounts(limit=limit)
        account_ids = [a.id for a in accounts if a.id]
    else:
        account_ids = None

    console.print("[blue]Extracting features...[/blue]")
    processed = extractor.extract_features_batch(account_ids=account_ids)
    console.print(f"[green]Extracted features for {processed} accounts[/green]")


@app.command()
def bot(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit accounts to score"),
):
    """Run bot detection."""
    detector = BotDetector()

    console.print("[blue]Running bot detection...[/blue]")

    # Get account IDs if limit specified
    if limit:
        db = get_db()
        accounts = db.get_all_accounts(limit=limit)
        account_ids = [a.id for a in accounts if a.id]
        scored = detector.score_all_accounts(account_ids=account_ids)
    else:
        scored = detector.score_all_accounts()

    console.print(f"[green]Scored {scored} accounts[/green]")

    # Show high bot score accounts
    high_bot = detector.get_high_confidence_bots(min_score=0.7, limit=10)
    if high_bot:
        console.print(f"\n[bold]Top suspected bots:[/bold]")
        for acc in high_bot:
            console.print(f"  @{acc.username}: {acc.bot_score:.2f}")


@app.command()
def stance(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit accounts to classify"),
    llm: bool = typer.Option(False, "--llm", help="Use LLM for classification (costs API credits)"),
    force: bool = typer.Option(False, "--force", "-f", help="Reclassify existing stances"),
):
    """Run stance classification."""
    classifier = StanceClassifier()

    if llm and not settings.anthropic_api_key:
        console.print("[red]Anthropic API key not configured[/red]")
        raise typer.Exit(1)

    classified = classifier.classify_all_accounts(use_llm=llm)
    console.print(f"[green]Classified {classified} accounts[/green]")

    # Show distribution
    db = get_db()
    accounts = db.get_all_accounts()
    stance_counts = {}
    for acc in accounts:
        s = acc.political_stance.value if acc.political_stance else "unknown"
        stance_counts[s] = stance_counts.get(s, 0) + 1

    console.print("\n[bold]Stance Distribution:[/bold]")
    for stance_name, count in sorted(stance_counts.items()):
        console.print(f"  {stance_name}: {count}")


@app.command()
def coordination(
    time_window: int = typer.Option(300, "--window", "-w", help="Time window in seconds"),
    similarity: float = typer.Option(0.7, "--similarity", "-s", help="Similarity threshold"),
):
    """Detect coordination patterns."""
    detector = CoordinationDetector()
    db = get_db()

    console.print("[blue]Running coordination detection...[/blue]")
    results = detector.detect_all()

    # Update coordination scores for involved accounts
    updated = detector.update_coordination_scores()

    # Show summary
    console.print(f"\n[bold]Coordination Detection Results:[/bold]")
    console.print(f"  Synchronized posting clusters: {results['synchronized_posting']}")
    console.print(f"  Hashtag campaign clusters: {results['hashtag_campaigns']}")
    console.print(f"  Amplification networks: {results['amplification_networks']}")
    console.print(f"  Total clusters: {sum(results.values())}")
    console.print(f"  Accounts updated: {updated}")

    # Show cluster details
    clusters = db.get_clusters()
    if clusters:
        console.print(f"\n[bold]Cluster Details:[/bold]")
        for cluster in clusters:
            console.print(f"\n  [{cluster.cluster_type.value}] {cluster.description}")
            console.print(f"    Confidence: {cluster.confidence_score:.2f}")
            console.print(f"    Members: {len(cluster.member_account_ids)} accounts")
            if cluster.evidence:
                for key, val in cluster.evidence.items():
                    console.print(f"    {key}: {val}")


@app.command("all")
def run_all(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit accounts"),
    force: bool = typer.Option(False, "--force", "-f", help="Recompute all"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM classification (use rules only)"),
):
    """Run full analysis pipeline."""
    console.print("[bold]Running full analysis pipeline...[/bold]\n")

    # 1. Feature extraction
    console.print("[blue]Step 1: Feature extraction[/blue]")
    extractor = FeatureExtractor()
    extractor.extract_features_batch()

    # 2. Bot detection
    console.print("\n[blue]Step 2: Bot detection[/blue]")
    detector = BotDetector()
    detector.score_all_accounts()

    # 3. Stance classification
    console.print("\n[blue]Step 3: Stance classification[/blue]")
    classifier = StanceClassifier()
    classifier.classify_all_accounts(use_llm=not no_llm)

    # 4. Coordination detection
    console.print("\n[blue]Step 4: Coordination detection[/blue]")
    coord_detector = CoordinationDetector()
    coord_detector.detect_all()

    console.print("\n[green]Analysis pipeline complete![/green]")

    # Final stats
    db = get_db()
    stats = db.get_stats()
    console.print("\n[bold]Final Statistics:[/bold]")
    for key, value in stats.items():
        console.print(f"  {key}: {value}")


@app.command()
def export(
    output: Path = typer.Argument(..., help="Output file path (.csv or .json)"),
    min_bot_score: float = typer.Option(None, "--min-bot", help="Minimum bot score filter"),
    stance: str = typer.Option(None, "--stance", help="Filter by stance"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit results"),
):
    """Export analysis results."""
    db = get_db()

    from seda.models import PoliticalStance

    stance_filter = None
    if stance:
        try:
            stance_filter = PoliticalStance(stance.lower())
        except ValueError:
            console.print(f"[red]Invalid stance: {stance}[/red]")
            raise typer.Exit(1)

    accounts = db.get_all_accounts(
        limit=limit,
        min_bot_score=min_bot_score,
        stance=stance_filter,
    )

    if not accounts:
        console.print("[yellow]No accounts match the filters[/yellow]")
        return

    # Prepare data
    data = []
    for account in accounts:
        data.append({
            "username": account.username,
            "display_name": account.display_name,
            "twitter_id": account.twitter_id,
            "followers": account.followers_count,
            "following": account.following_count,
            "tweets": account.tweet_count,
            "bot_score": account.bot_score,
            "regime_score": account.regime_score,
            "coordination_score": account.coordination_score,
            "account_type": account.account_type.value,
            "political_stance": account.political_stance.value,
            "is_seed": account.is_seed,
            "created_at": account.created_at.isoformat() if account.created_at else None,
        })

    # Export based on file extension
    suffix = output.suffix.lower()

    if suffix == ".json":
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    elif suffix == ".csv":
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(output, index=False)
    else:
        console.print(f"[red]Unsupported format: {suffix}. Use .csv or .json[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Exported {len(data)} accounts to {output}[/green]")


@app.command()
def report():
    """Generate analysis report."""
    db = get_db()
    stats = db.get_stats()

    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]        SEDA Analysis Report           [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

    # Overview
    console.print("[bold]Overview[/bold]")
    console.print(f"  Total accounts: {stats['accounts']}")
    console.print(f"  Seed accounts: {stats['seeds']}")
    console.print(f"  Total tweets: {stats['tweets']}")
    console.print(f"  Coordination clusters: {stats['clusters']}")

    # Bot detection
    accounts = db.get_all_accounts()
    scored = [a for a in accounts if a.bot_score is not None]
    high_bot = [a for a in accounts if a.bot_score and a.bot_score >= 0.7]
    console.print("\n[bold]Bot Detection[/bold]")
    console.print(f"  Accounts scored: {len(scored)}")
    console.print(f"  High bot score (>=0.7): {len(high_bot)}")

    # Stance distribution
    console.print("\n[bold]Stance Distribution[/bold]")
    stance_counts = {}
    for a in accounts:
        stance = a.political_stance.value if a.political_stance else "unknown"
        stance_counts[stance] = stance_counts.get(stance, 0) + 1
    for stance, count in sorted(stance_counts.items()):
        pct = (count / max(len(accounts), 1)) * 100
        console.print(f"  {stance}: {count} ({pct:.1f}%)")

    # Coordination
    clusters = db.get_clusters()
    involved_accounts = set()
    for c in clusters:
        involved_accounts.update(c.member_account_ids)
    console.print("\n[bold]Coordination Detection[/bold]")
    console.print(f"  Total clusters: {len(clusters)}")
    console.print(f"  Accounts involved: {len(involved_accounts)}")

    # High-risk accounts
    console.print("\n[bold]High-Risk Accounts (bot score >= 0.7)[/bold]")
    if high_bot:
        table = Table()
        table.add_column("Username")
        table.add_column("Bot Score")
        table.add_column("Stance")
        table.add_column("Coordination")

        for acc in sorted(high_bot, key=lambda x: x.bot_score or 0, reverse=True)[:10]:
            table.add_row(
                f"@{acc.username}",
                f"{acc.bot_score:.2f}" if acc.bot_score else "N/A",
                acc.political_stance.value if acc.political_stance else "N/A",
                f"{acc.coordination_score:.2f}" if acc.coordination_score else "N/A",
            )
        console.print(table)
    else:
        console.print("  No high-risk accounts found")

    console.print("\n[dim]Report generated at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "[/dim]")


@app.command()
def explain(
    username: str = typer.Argument(..., help="Username to explain"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM for detailed explanation"),
):
    """Explain analysis scores and threat classification for an account."""
    db = get_db()

    username = username.lstrip("@")
    account = db.get_account_by_username(username)

    if not account:
        console.print(f"[red]Account not found: @{username}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Analysis Explanation for @{username}[/bold]\n")

    # Basic info
    console.print("[bold]Profile[/bold]")
    console.print(f"  Display name: {account.display_name or 'N/A'}")
    console.print(f"  Followers: {account.followers_count:,}")
    console.print(f"  Following: {account.following_count:,}")
    console.print(f"  Tweets: {account.tweet_count:,}")
    console.print(f"  Account type: {account.account_type.value}")
    console.print(f"  Is seed: {account.is_seed}")

    # Scores
    console.print("\n[bold]Scores[/bold]")
    console.print(f"  Bot score: {account.bot_score:.3f}" if account.bot_score else "  Bot score: N/A")
    console.print(f"  Regime score: {account.regime_score:.3f}" if account.regime_score else "  Regime score: N/A")
    console.print(f"  Coordination score: {account.coordination_score:.3f}" if account.coordination_score else "  Coordination score: N/A")
    console.print(f"  Political stance: {account.political_stance.value}")
    console.print(f"  Political taxonomy: {account.political_taxonomy.value}")
    console.print(f"  [bold]Threat level: {account.threat_level.value if account.threat_level else 'unknown'}[/bold]")

    # Threat level explanation
    console.print("\n[bold]Threat Classification Explanation[/bold]")
    classifier = StanceClassifier()
    explanation = classifier.explain_account(account)

    console.print(f"  Activity status: {explanation['activity_status']}")
    console.print(f"\n  [cyan]Signal counts:[/cyan]")
    console.print(f"    Violence keywords: {explanation['signal_counts']['violence']}")
    console.print(f"    IRGC references: {explanation['signal_counts']['irgc']}")
    console.print(f"    Doxxing indicators: {explanation['signal_counts']['doxxing']}")
    console.print(f"    Harassment keywords: {explanation['signal_counts']['harassment']}")

    console.print(f"\n  [cyan]Rule-based explanation:[/cyan]")
    console.print(f"    {explanation['rule_based_explanation']}")

    # Show example tweets if available
    for signal_type, examples in explanation['examples'].items():
        if examples:
            console.print(f"\n  [yellow]Example {signal_type} tweets:[/yellow]")
            for ex in examples[:2]:
                console.print(f"    [{ex['date']}] {ex['text'][:150]}...")
                console.print(f"    Keywords: {', '.join(ex['keywords'])}")

    # LLM explanation
    if use_llm and 'llm_explanation' in explanation:
        console.print(f"\n  [green]LLM Analysis:[/green]")
        console.print(f"    {explanation['llm_explanation']}")
    elif use_llm:
        console.print("\n  [yellow]LLM explanation not available (check ANTHROPIC_API_KEY)[/yellow]")

    # Bot score explanation
    if account.bot_score is not None:
        console.print("\n[bold]Bot Score Explanation[/bold]")
        try:
            detector = BotDetector()
            bot_explanation = detector.explain_prediction(account)

            console.print(f"  Prediction: {'BOT' if bot_explanation['is_bot'] else 'HUMAN'}")
            console.print(f"  Confidence: {bot_explanation['bot_score']:.3f}")

            console.print("\n  Top contributing factors:")
            for name, info in list(bot_explanation.get("top_factors", {}).items())[:5]:
                console.print(f"    {name}: {info['value']:.3f} (importance: {info['importance']:.3f})")
        except Exception as e:
            console.print(f"  [yellow]Could not explain: {e}[/yellow]")

    # Recent tweets
    tweets = db.get_tweets_by_account(account.id, limit=5)
    if tweets:
        console.print("\n[bold]Recent Tweets[/bold]")
        for tweet in tweets:
            text = tweet.text[:100] + "..." if len(tweet.text) > 100 else tweet.text
            date = tweet.created_at.strftime("%Y-%m-%d") if tweet.created_at else "Unknown"
            console.print(f"  [{date}] {text}")


@app.command("train-bot")
def train_bot(
    use_weak_labels: bool = typer.Option(True, "--weak", help="Use weak supervision labels"),
    cv_folds: int = typer.Option(5, "--cv", help="Cross-validation folds"),
):
    """Train bot detection model using LightGBM."""
    detector = BotDetector()

    console.print("[blue]Training bot detection model...[/blue]")

    if use_weak_labels:
        console.print("Creating weak supervision labels...")
        positive_ids, negative_ids = detector.create_weak_labels()
        console.print(f"  Found {len(positive_ids)} likely bots")
        console.print(f"  Found {len(negative_ids)} likely humans")

        if len(positive_ids) < 5 or len(negative_ids) < 5:
            console.print("[red]Not enough training data. Collect more accounts first.[/red]")
            raise typer.Exit(1)

        # Cross-validation
        if cv_folds > 1:
            console.print(f"\nRunning {cv_folds}-fold cross-validation...")
            cv_results = detector.cross_validate(positive_ids, negative_ids, n_folds=cv_folds)
            if "mean_auc" in cv_results:
                console.print(f"  Mean AUC: {cv_results['mean_auc']:.3f} (+/- {cv_results['std_auc']:.3f})")
            else:
                console.print("  [yellow]Cross-validation failed[/yellow]")

        # Train final model
        result = detector.train_from_weak_labels()
        if result:
            console.print(f"\n[green]Model trained successfully![/green]")
            console.print(f"  Final AUC: {result.get('auc', 'N/A')}")
            console.print(f"  Model saved to: data/models/bot_classifier.lgb")
        else:
            console.print("[red]Training failed[/red]")
    else:
        console.print("[yellow]Manual labels not supported yet. Use --weak[/yellow]")


@app.command("embed")
def embed(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit accounts to embed"),
    tweets: bool = typer.Option(False, "--tweets", help="Also embed tweets"),
):
    """Generate ParsBERT embeddings for accounts/tweets."""
    if not is_embeddings_available():
        console.print("[red]Embeddings require transformers and torch.[/red]")
        console.print("Install with: pip install transformers torch")
        raise typer.Exit(1)

    from seda.analysis.embeddings import PersianEmbedder

    console.print("[blue]Initializing ParsBERT model...[/blue]")
    embedder = PersianEmbedder()
    console.print(f"  Using device: {embedder.device}")

    console.print("\n[blue]Generating account embeddings...[/blue]")
    db = get_db()

    if limit:
        accounts = db.get_all_accounts(limit=limit)
        account_ids = [a.id for a in accounts if a.id]
    else:
        account_ids = None

    embedded = embedder.embed_all_accounts(account_ids=account_ids)
    console.print(f"[green]Embedded {embedded} accounts[/green]")

    if tweets:
        console.print("\n[blue]Generating tweet embeddings...[/blue]")
        tweet_count = embedder.embed_all_tweets(limit=limit or 10000)
        console.print(f"[green]Embedded {tweet_count} tweets[/green]")


@app.command("threat")
def threat_report():
    """Generate threat level report for pro-regime accounts."""
    db = get_db()

    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]     Pro-Regime Threat Level Report    [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

    # Get threat stats
    threat_stats = db.get_threat_level_stats()

    if not threat_stats:
        console.print("[yellow]No threat levels classified yet. Run stance analysis first.[/yellow]")
        return

    # Display breakdown
    total = sum(threat_stats.values())
    console.print(f"[bold]Total Pro-Regime Accounts: {total}[/bold]\n")

    table = Table()
    table.add_column("Threat Level", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Risk", justify="center")

    # Risk indicators
    risk_levels = {
        "violence_inciter": ("CRITICAL", "red"),
        "doxxer": ("CRITICAL", "red"),
        "irgc_operative": ("HIGH", "yellow"),
        "state_propagandist": ("HIGH", "yellow"),
        "amplifier_bot": ("MEDIUM", "cyan"),
        "troll": ("MEDIUM", "cyan"),
        "passive_supporter": ("LOW", "green"),
        "unknown": ("UNKNOWN", "dim"),
    }

    for level, count in sorted(threat_stats.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100 if total > 0 else 0
        risk, color = risk_levels.get(level, ("UNKNOWN", "dim"))
        table.add_row(
            level.replace("_", " ").title(),
            str(count),
            f"{pct:.1f}%",
            f"[{color}]{risk}[/{color}]",
        )

    console.print(table)

    # High priority targets
    console.print("\n[bold]High-Priority Targets (Violence Inciters + Doxxers):[/bold]")

    high_priority = []
    for level in [ThreatLevel.VIOLENCE_INCITER, ThreatLevel.DOXXER]:
        accounts = db.get_accounts_by_threat_level(level, limit=10)
        high_priority.extend(accounts)

    if high_priority:
        for acc in high_priority[:15]:
            console.print(f"  @{acc.username} - {acc.threat_level.value}")
    else:
        console.print("  None identified yet")

    console.print("\n[dim]Report generated at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "[/dim]")


@app.command("similarity")
def find_similar(
    username: str = typer.Argument(..., help="Username to find similar accounts for"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of similar accounts"),
):
    """Find accounts similar to a given account using embeddings."""
    if not is_embeddings_available():
        console.print("[red]Embeddings require transformers and torch.[/red]")
        console.print("Install with: pip install transformers torch")
        raise typer.Exit(1)

    from seda.analysis.embeddings import PersianEmbedder

    db = get_db()
    username = username.lstrip("@")
    account = db.get_account_by_username(username)

    if not account:
        console.print(f"[red]Account not found: @{username}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Finding accounts similar to @{username}...[/blue]")

    embedder = PersianEmbedder()

    similar = embedder.find_similar_accounts(account, top_k=top_k)

    if not similar:
        console.print("[yellow]No similar accounts found. Generate embeddings first.[/yellow]")
        return

    console.print(f"\n[bold]Top {len(similar)} Similar Accounts:[/bold]")

    table = Table()
    table.add_column("Username")
    table.add_column("Similarity", justify="right")
    table.add_column("Stance")
    table.add_column("Threat Level")

    for sim_acc, score in similar:
        table.add_row(
            f"@{sim_acc.username}",
            f"{score:.3f}",
            sim_acc.political_stance.value,
            sim_acc.threat_level.value if sim_acc.threat_level else "unknown",
        )

    console.print(table)


if __name__ == "__main__":
    app()
