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
)

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
):
    """Explain analysis scores for an account."""
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

    # Bot score explanation
    if account.bot_score is not None:
        console.print("\n[bold]Bot Score Explanation[/bold]")
        try:
            detector = BotDetector()
            explanation = detector.explain_prediction(account)

            console.print(f"  Prediction: {'BOT' if explanation['is_bot'] else 'HUMAN'}")
            console.print(f"  Confidence: {explanation['bot_score']:.3f}")

            console.print("\n  Top contributing factors:")
            for name, info in list(explanation.get("top_factors", {}).items())[:5]:
                console.print(f"    {name}: {info['value']:.3f} (importance: {info['importance']:.3f})")
        except Exception as e:
            console.print(f"  [yellow]Could not explain: {e}[/yellow]")

    # Recent tweets
    tweets = db.get_tweets_by_account(account.id, limit=5)
    if tweets:
        console.print("\n[bold]Recent Tweets[/bold]")
        for tweet in tweets:
            text = tweet.text[:100] + "..." if len(tweet.text) > 100 else tweet.text
            console.print(f"  - {text}")


if __name__ == "__main__":
    app()
