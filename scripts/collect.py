"""Data collection CLI."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress

from seda.config import get_settings

settings = get_settings()
from seda.db import get_db
from seda.models import Account, AccountType, SeedCategory
from seda.scraper import TwitterScraper

app = typer.Typer(help="Collect Twitter data using Apify")
console = Console()


def _get_scraper() -> TwitterScraper:
    """Get scraper instance with API validation."""
    if not settings.apify_api_token:
        console.print("[red]Apify API token not configured. Set APIFY_API_TOKEN in .env[/red]")
        raise typer.Exit(1)
    return TwitterScraper()


@app.command()
def account(
    username: str = typer.Argument(..., help="Twitter username to scrape"),
    tweets: int = typer.Option(100, "--tweets", "-t", help="Number of tweets to fetch"),
):
    """Scrape a single account and its tweets."""
    db = get_db()
    scraper = _get_scraper()

    username = username.lstrip("@")
    console.print(f"[blue]Scraping @{username}...[/blue]")

    account_obj, tweets_list, result = scraper.scrape_user_with_tweets(username, max_tweets=tweets)

    if not result.success:
        console.print(f"[red]Scraper error: {result.error}[/red]")
        raise typer.Exit(1)

    if account_obj:
        account_id = db.insert_account(account_obj)
        console.print(f"[green]Saved account: @{username} (ID: {account_id})[/green]")

        if tweets_list:
            for tweet in tweets_list:
                tweet.account_id = account_id
            db.insert_tweets_bulk(tweets_list)
            console.print(f"[green]Saved {len(tweets_list)} tweets[/green]")
    else:
        console.print(f"[red]Failed to scrape @{username}[/red]")


@app.command()
def seeds(
    tweets: int = typer.Option(100, "--tweets", "-t", help="Tweets per account"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by seed category"),
):
    """Scrape all seed accounts."""
    db = get_db()
    scraper = _get_scraper()

    cat_filter = None
    if category:
        try:
            cat_filter = SeedCategory(category.lower())
        except ValueError:
            console.print(f"[red]Invalid category[/red]")
            raise typer.Exit(1)

    seed_accounts = db.get_seeds(category=cat_filter)

    if not seed_accounts:
        console.print("[yellow]No seed accounts found. Run 'seda-seeds init' first.[/yellow]")
        return

    console.print(f"[blue]Scraping {len(seed_accounts)} seed accounts...[/blue]")

    success = 0
    failed = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Scraping seeds...", total=len(seed_accounts))

        for seed in seed_accounts:
            try:
                account_obj, tweets_list, result = scraper.scrape_user_with_tweets(
                    seed.username, max_tweets=tweets
                )

                if result.success and account_obj:
                    account_obj.is_seed = True
                    account_obj.seed_category = seed.category

                    # Set account type based on seed category
                    if seed.category == SeedCategory.OFFICIAL_STATE:
                        account_obj.account_type = AccountType.OFFICIAL_STATE
                    elif seed.category == SeedCategory.STATE_MEDIA:
                        account_obj.account_type = AccountType.STATE_MEDIA
                    elif seed.category == SeedCategory.IRGC_LINKED:
                        account_obj.account_type = AccountType.IRGC_LINKED

                    account_id = db.insert_account(account_obj)

                    if tweets_list:
                        for tweet in tweets_list:
                            tweet.account_id = account_id
                        db.insert_tweets_bulk(tweets_list)

                    success += 1
                else:
                    failed.append(seed.username)

            except Exception as e:
                console.print(f"[yellow]Error scraping @{seed.username}: {e}[/yellow]")
                failed.append(seed.username)

            progress.update(task, advance=1)

    console.print(f"\n[green]Successfully scraped {success}/{len(seed_accounts)} seed accounts[/green]")
    if failed:
        console.print(f"[yellow]Failed: {', '.join(failed)}[/yellow]")


@app.command()
def expand(
    depth: int = typer.Option(1, "--depth", "-d", help="Expansion depth (levels of retweeters)"),
    tweets_per_seed: int = typer.Option(10, "--tweets", "-t", help="Tweets per seed to find retweeters for"),
    retweeters: int = typer.Option(50, "--retweeters", "-r", help="Retweeters per tweet"),
):
    """Expand network by finding retweeters of seed tweets."""
    db = get_db()
    scraper = _get_scraper()

    seed_accounts = db.get_all_accounts(is_seed=True)

    if not seed_accounts:
        console.print("[yellow]No seed accounts found. Collect seeds first.[/yellow]")
        return

    console.print(f"[blue]Expanding network from {len(seed_accounts)} seed accounts...[/blue]")

    new_accounts = 0

    for level in range(depth):
        console.print(f"\n[bold]Expansion level {level + 1}/{depth}[/bold]")

        for seed in seed_accounts:
            tweets = db.get_tweets_by_account(seed.id, limit=tweets_per_seed)

            if not tweets:
                continue

            console.print(f"[blue]Finding retweeters for @{seed.username}...[/blue]")

            for tweet in tweets:
                if tweet.retweet_count == 0:
                    continue

                try:
                    retweeter_accounts, result = scraper.scrape_tweet_retweeters(
                        tweet.tweet_id, max_items=retweeters
                    )

                    if result.success and retweeter_accounts:
                        for rt_account in retweeter_accounts:
                            existing = db.get_account_by_twitter_id(rt_account.twitter_id)
                            if not existing:
                                rt_account.account_type = AccountType.AMPLIFIER
                                db.insert_account(rt_account)
                                new_accounts += 1
                        console.print(f"  Found {len(retweeter_accounts)} retweeters for tweet {tweet.tweet_id[:8]}...")

                except Exception as e:
                    console.print(f"[yellow]Error getting retweeters: {e}[/yellow]")

    console.print(f"\n[green]Added {new_accounts} new accounts through expansion[/green]")


@app.command()
def followers(
    username: str = typer.Argument(None, help="Username to get followers from (or all seeds if not specified)"),
    max_followers: int = typer.Option(500, "--max", "-m", help="Max followers per account"),
    all_seeds: bool = typer.Option(False, "--all-seeds", "-a", help="Get followers from all seed accounts"),
):
    """Get followers of seed accounts to find potential amplifiers."""
    db = get_db()
    scraper = _get_scraper()

    if username:
        usernames = [username.lstrip("@")]
    elif all_seeds:
        seed_accounts = db.get_all_accounts(is_seed=True)
        usernames = [s.username for s in seed_accounts]
    else:
        console.print("[red]Specify a username or use --all-seeds[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Getting followers from {len(usernames)} account(s)...[/blue]")

    new_accounts = 0
    with Progress() as progress:
        task = progress.add_task("[cyan]Scraping followers...", total=len(usernames))

        for uname in usernames:
            try:
                follower_accounts, result = scraper.scrape_user_followers(uname, max_items=max_followers)

                if result.success and follower_accounts:
                    console.print(f"  @{uname}: found {len(follower_accounts)} followers")
                    for account in follower_accounts:
                        existing = db.get_account_by_twitter_id(account.twitter_id)
                        if not existing:
                            account.account_type = AccountType.AMPLIFIER
                            db.insert_account(account)
                            new_accounts += 1
                else:
                    console.print(f"  [yellow]@{uname}: {result.error or 'no followers found'}[/yellow]")

            except Exception as e:
                console.print(f"[yellow]Error getting followers for @{uname}: {e}[/yellow]")

            progress.update(task, advance=1)

    console.print(f"\n[green]Added {new_accounts} new accounts from followers[/green]")


@app.command()
def batch(
    file: Path = typer.Argument(..., help="File with usernames (one per line)"),
    tweets: int = typer.Option(100, "--tweets", "-t", help="Tweets per account"),
):
    """Scrape accounts from a file."""
    db = get_db()
    scraper = _get_scraper()

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    usernames = [
        line.strip().lstrip("@")
        for line in file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    console.print(f"[blue]Scraping {len(usernames)} accounts from {file}...[/blue]")

    success = 0

    with Progress() as progress:
        task = progress.add_task("[cyan]Scraping...", total=len(usernames))

        for username in usernames:
            try:
                account_obj, tweets_list, result = scraper.scrape_user_with_tweets(
                    username, max_tweets=tweets
                )

                if result.success and account_obj:
                    account_id = db.insert_account(account_obj)
                    if tweets_list:
                        for tweet in tweets_list:
                            tweet.account_id = account_id
                        db.insert_tweets_bulk(tweets_list)
                    success += 1

            except Exception as e:
                console.print(f"[yellow]Error scraping @{username}: {e}[/yellow]")

            progress.update(task, advance=1)

    console.print(f"\n[green]Successfully scraped {success}/{len(usernames)} accounts[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    max_tweets: int = typer.Option(100, "--max", "-m", help="Maximum tweets to fetch"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save found accounts to database"),
):
    """Search for tweets by query and optionally save accounts."""
    db = get_db()
    scraper = _get_scraper()

    console.print(f"[blue]Searching for: {query}[/blue]")

    tweets, accounts, result = scraper.scrape_search(query, max_items=max_tweets)

    if not result.success:
        console.print(f"[red]Search failed: {result.error}[/red]")
        return

    new_accounts = 0
    if save and accounts:
        console.print(f"[blue]Saving {len(accounts)} accounts...[/blue]")
        for account in accounts:
            existing = db.get_account_by_twitter_id(account.twitter_id)
            if not existing:
                db.insert_account(account)
                new_accounts += 1

    console.print(f"[green]Found {len(tweets)} tweets from {len(accounts)} unique accounts[/green]")
    if save:
        console.print(f"[green]Added {new_accounts} new accounts to database[/green]")


@app.command()
def usage():
    """Show Apify API usage statistics."""
    scraper = _get_scraper()
    stats = scraper.get_usage_stats()

    console.print("\n[bold]Apify Usage Statistics[/bold]\n")
    for key, value in stats.items():
        console.print(f"  {key}: {value}")


@app.command()
def stats():
    """Show collection statistics."""
    db = get_db()
    db_stats = db.get_stats()

    console.print("\n[bold]Collection Statistics[/bold]\n")
    for key, value in db_stats.items():
        console.print(f"  {key}: {value}")


if __name__ == "__main__":
    app()
