"""Seed account management CLI."""

import typer
from rich.console import Console
from rich.table import Table

from seda.db import get_db
from seda.models import SeedAccount, SeedCategory

app = typer.Typer(help="Manage seed accounts for network expansion")
console = Console()

# Default seed accounts based on research (~50 accounts)
# Note: Some accounts may be suspended/renamed - scraper handles gracefully
DEFAULT_SEEDS = [
    # ==================== OFFICIAL STATE ====================
    # Supreme Leader Office
    ("khamenei_ir", SeedCategory.OFFICIAL_STATE, "Supreme Leader Ayatollah Khamenei (English)"),
    ("Khamenei_fa", SeedCategory.OFFICIAL_STATE, "Supreme Leader Ayatollah Khamenei (Farsi)"),
    ("Khamenei_ar", SeedCategory.OFFICIAL_STATE, "Supreme Leader Ayatollah Khamenei (Arabic)"),
    ("Khamenei_es", SeedCategory.OFFICIAL_STATE, "Supreme Leader Ayatollah Khamenei (Spanish)"),

    # President & Government
    ("drpezeshkian", SeedCategory.OFFICIAL_STATE, "President Masoud Pezeshkian"),
    ("Iran_GOV", SeedCategory.OFFICIAL_STATE, "Government of Islamic Republic of Iran"),
    ("raisi_com", SeedCategory.OFFICIAL_STATE, "Late President Ebrahim Raisi"),
    ("HassanRouhani", SeedCategory.OFFICIAL_STATE, "Former President Hassan Rouhani"),
    ("Rouhani_ir", SeedCategory.OFFICIAL_STATE, "Former President Rouhani (Farsi)"),
    ("Ahmadinejad1956", SeedCategory.OFFICIAL_STATE, "Former President Mahmoud Ahmadinejad"),

    # Parliament
    ("mb_ghalibaf", SeedCategory.OFFICIAL_STATE, "Parliament Speaker Mohammad Bagher Ghalibaf"),
    ("alilarijani_ir", SeedCategory.OFFICIAL_STATE, "Ali Larijani - SNSC Secretary, ex-Parliament Speaker"),
    ("mah_sadeghi", SeedCategory.OFFICIAL_STATE, "MP Mahmoud Sadeghi (reformist)"),

    # Foreign Ministry & Diplomats
    ("araghchi", SeedCategory.OFFICIAL_STATE, "Foreign Minister Abbas Araghchi"),
    ("JZarif", SeedCategory.OFFICIAL_STATE, "Former FM Mohammad Javad Zarif"),
    ("Amirabdolahian", SeedCategory.OFFICIAL_STATE, "Late FM Hossein Amirabdollahian"),
    ("TakhtRavanchi", SeedCategory.OFFICIAL_STATE, "Deputy FM Majid Takht-Ravanchi"),
    ("Gharibabadi", SeedCategory.OFFICIAL_STATE, "Deputy FM Kazem Gharibabadi"),
    ("IRIMFA_EN", SeedCategory.OFFICIAL_STATE, "Ministry of Foreign Affairs (English)"),

    # Security Council & Military Officials
    ("DrSaeedJalili", SeedCategory.OFFICIAL_STATE, "Saeed Jalili - Supreme Leader Rep to SNSC"),

    # Other Officials
    ("azarijahromi", SeedCategory.OFFICIAL_STATE, "Former IT Minister Azari Jahromi"),
    ("ebtekarm", SeedCategory.OFFICIAL_STATE, "Massoumeh Ebtekar - former VP Environment"),
    ("mowlaverdi", SeedCategory.OFFICIAL_STATE, "Shahindokht Molaverdi - former VP Women Affairs"),

    # ==================== STATE MEDIA ====================
    # IRIB (State Broadcaster) Channels
    ("PressTV", SeedCategory.STATE_MEDIA, "Press TV English (IRIB)"),
    ("AlAlamTV", SeedCategory.STATE_MEDIA, "Al Alam TV - Arabic (IRIB)"),
    ("HispanTV", SeedCategory.STATE_MEDIA, "HispanTV - Spanish (IRIB)"),

    # News Agencies - IRNA
    ("IrnaEnglish", SeedCategory.STATE_MEDIA, "IRNA English"),

    # News Agencies - Tasnim (IRGC-affiliated)
    ("tasnimnews_en", SeedCategory.STATE_MEDIA, "Tasnim News Agency (English)"),
    ("Tasnimnews", SeedCategory.STATE_MEDIA, "Tasnim News (Farsi)"),

    # News Agencies - Fars (IRGC-affiliated)
    ("FarsNews_Agency", SeedCategory.STATE_MEDIA, "Fars News (Persian)"),
    ("EnglishFars", SeedCategory.STATE_MEDIA, "Fars News (English)"),

    # News Agencies - Others
    ("MehrnewsCom", SeedCategory.STATE_MEDIA, "Mehr News Agency"),

    # Newspapers
    ("TehranTimes79", SeedCategory.STATE_MEDIA, "Tehran Times"),

    # ==================== EMBASSIES & MISSIONS ====================
    ("Iran_UN", SeedCategory.OFFICIAL_STATE, "Iran Mission to UN, New York"),
    ("Iran_in_UK", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in UK"),
    ("IranEmbassyJP", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in Japan"),
    ("IraninRussia", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in Russia"),
    ("IranEmbassyTR", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in Turkey"),
    ("IranEmbPak", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in Pakistan"),
    ("IranEmbassyDE", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in Germany"),
    ("IranEmbFR", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in France"),
    ("IranEmbassyCN", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in China"),
    ("IranEmbassyIN", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in India"),
    ("IranEmbassyIRQ", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in Iraq"),
    ("IranEmbassySY", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in Syria"),
    ("IranEmbassyLB", SeedCategory.OFFICIAL_STATE, "Iranian Embassy in Lebanon"),

    # ==================== IRGC-LINKED ====================
    # Note: Tasnim & Fars (listed above) are IRGC-affiliated
    # Many IRGC commander accounts have been suspended

    # ==================== OTHER REGIME-AFFILIATED ====================
    ("Khatamimedia", SeedCategory.OTHER, "Former President Mohammad Khatami"),
    ("mohirezaee", SeedCategory.OTHER, "Mohsen Rezaei - ex-IRGC commander, Expediency Council"),
]


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing seeds"),
):
    """Initialize default seed accounts."""
    db = get_db()

    existing = db.get_seeds()
    if existing and not force:
        console.print(f"[yellow]Found {len(existing)} existing seed accounts. Use --force to overwrite.[/yellow]")
        return

    count = 0
    for username, category, notes in DEFAULT_SEEDS:
        seed = SeedAccount(
            username=username,
            category=category,
            notes=notes,
        )
        db.insert_seed(seed)
        count += 1

    console.print(f"[green]Initialized {count} seed accounts[/green]")


@app.command()
def add(
    username: str = typer.Argument(..., help="Twitter username (without @)"),
    category: str = typer.Argument(..., help="Category: official_state, state_media, irgc_linked, other"),
    notes: str = typer.Option("", "--notes", "-n", help="Optional notes"),
):
    """Add a new seed account."""
    db = get_db()

    # Parse category
    try:
        cat = SeedCategory(category.lower())
    except ValueError:
        console.print(f"[red]Invalid category. Use: official_state, state_media, irgc_linked, other[/red]")
        raise typer.Exit(1)

    seed = SeedAccount(
        username=username.lstrip("@"),
        category=cat,
        notes=notes,
    )

    db.insert_seed(seed)
    console.print(f"[green]Added seed account: @{username}[/green]")


@app.command("list")
def list_seeds(
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
):
    """List all seed accounts."""
    db = get_db()

    cat_filter = None
    if category:
        try:
            cat_filter = SeedCategory(category.lower())
        except ValueError:
            console.print(f"[red]Invalid category[/red]")
            raise typer.Exit(1)

    seeds = db.get_seeds(category=cat_filter)

    if not seeds:
        console.print("[yellow]No seed accounts found[/yellow]")
        return

    table = Table(title="Seed Accounts")
    table.add_column("Username", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Added", style="dim")
    table.add_column("Notes")

    for seed in seeds:
        table.add_row(
            f"@{seed.username}",
            seed.category.value,
            seed.added_at.strftime("%Y-%m-%d"),
            seed.notes or "",
        )

    console.print(table)
    console.print(f"\nTotal: {len(seeds)} seed accounts")


@app.command()
def remove(
    username: str = typer.Argument(..., help="Twitter username to remove"),
):
    """Remove a seed account."""
    db = get_db()

    username = username.lstrip("@")
    if db.remove_seed(username):
        console.print(f"[green]Removed seed account: @{username}[/green]")
    else:
        console.print(f"[yellow]Seed account not found: @{username}[/yellow]")


@app.command()
def stats():
    """Show seed account statistics."""
    db = get_db()

    seeds = db.get_seeds()

    by_category = {}
    for seed in seeds:
        cat = seed.category.value
        by_category[cat] = by_category.get(cat, 0) + 1

    console.print("\n[bold]Seed Account Statistics[/bold]\n")
    console.print(f"Total: {len(seeds)}")
    for cat, count in sorted(by_category.items()):
        console.print(f"  {cat}: {count}")


if __name__ == "__main__":
    app()
