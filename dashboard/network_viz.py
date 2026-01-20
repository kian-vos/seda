"""Network visualization for SEDA dashboard."""

from typing import Optional

import plotly.graph_objects as go

from seda.db import get_db
from seda.models import Account, PoliticalStance, ThreatLevel

# Try to import networkx
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


def create_retweet_network(
    accounts: list[Account],
    max_nodes: int = 500,
    layout: str = "spring",
) -> Optional[go.Figure]:
    """Build directed graph of retweet relationships.

    Args:
        accounts: List of accounts to include
        max_nodes: Maximum number of nodes for performance
        layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')

    Returns:
        Plotly figure or None if networkx not available
    """
    if not NETWORKX_AVAILABLE:
        return None

    db = get_db()
    G = nx.DiGraph()

    # Add nodes (accounts)
    account_map = {a.twitter_id: a for a in accounts[:max_nodes]}

    for acc in accounts[:max_nodes]:
        G.add_node(
            acc.twitter_id,
            username=acc.username,
            stance=acc.political_stance.value,
            threat_level=acc.threat_level.value,
            bot_score=acc.bot_score or 0,
            followers=acc.followers_count,
        )

    # Add edges (retweet relationships)
    for acc in accounts[:max_nodes]:
        if not acc.id:
            continue

        tweets = db.get_tweets_by_account(acc.id, limit=100)
        for tweet in tweets:
            if tweet.is_retweet and tweet.referenced_tweet_id:
                # Find the original author if in our account set
                # Note: This is simplified - full implementation would track retweet sources
                pass

    # Create layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai" and len(G.nodes()) < 200:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.5)

    return _create_plotly_network(G, pos, "Retweet Network")


def create_coordination_network(
    clusters: list,
    accounts: list[Account],
    max_nodes: int = 500,
) -> Optional[go.Figure]:
    """Visualize coordination clusters as a network.

    Args:
        clusters: List of CoordinationCluster objects
        accounts: List of accounts
        max_nodes: Maximum number of nodes

    Returns:
        Plotly figure or None if networkx not available
    """
    if not NETWORKX_AVAILABLE:
        return None

    G = nx.Graph()
    account_map = {a.id: a for a in accounts}

    # Add nodes and edges from clusters
    for cluster in clusters:
        members = cluster.member_account_ids[:max_nodes]

        for member_id in members:
            acc = account_map.get(member_id)
            if acc:
                G.add_node(
                    member_id,
                    username=acc.username,
                    stance=acc.political_stance.value,
                    threat_level=acc.threat_level.value,
                    bot_score=acc.bot_score or 0,
                    cluster_type=cluster.cluster_type.value,
                )

        # Connect all members within the cluster
        for i, m1 in enumerate(members):
            for m2 in members[i+1:]:
                if m1 in G.nodes() and m2 in G.nodes():
                    G.add_edge(m1, m2, cluster_type=cluster.cluster_type.value)

    if len(G.nodes()) == 0:
        return None

    pos = nx.spring_layout(G, k=0.8, iterations=50)
    return _create_plotly_network(G, pos, "Coordination Network")


def create_threat_network(
    accounts: list[Account],
    max_nodes: int = 500,
) -> Optional[go.Figure]:
    """Create network visualization colored by threat level.

    Args:
        accounts: List of accounts (should be pro-regime)
        max_nodes: Maximum number of nodes

    Returns:
        Plotly figure or None if networkx not available
    """
    if not NETWORKX_AVAILABLE:
        return None

    G = nx.Graph()

    # Filter to pro-regime accounts only
    pro_regime = [a for a in accounts if a.political_stance == PoliticalStance.PRO_REGIME]

    for acc in pro_regime[:max_nodes]:
        G.add_node(
            acc.twitter_id,
            username=acc.username,
            threat_level=acc.threat_level.value,
            bot_score=acc.bot_score or 0,
            followers=acc.followers_count,
        )

    if len(G.nodes()) == 0:
        return None

    pos = nx.spring_layout(G, k=0.5)
    return _create_plotly_network_by_threat(G, pos, "Threat Level Network")


def _create_plotly_network(
    G: "nx.Graph",
    pos: dict,
    title: str,
) -> go.Figure:
    """Convert NetworkX graph to Plotly figure.

    Args:
        G: NetworkX graph
        pos: Node positions dict
        title: Chart title

    Returns:
        Plotly figure
    """
    # Edge traces
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#555555'),
        hoverinfo='none',
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    # Color map for stance
    stance_colors = {
        "pro_regime": "#dc143c",
        "anti_regime": "#2ecc71",
        "neutral": "#888888",
        "unknown": "#555555",
    }

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        data = G.nodes[node]
        username = data.get("username", str(node))
        stance = data.get("stance", "unknown")
        bot_score = data.get("bot_score", 0)
        followers = data.get("followers", 0)

        node_text.append(f"@{username}<br>Stance: {stance}<br>Bot: {bot_score:.2f}<br>Followers: {followers}")
        node_colors.append(stance_colors.get(stance, "#555555"))
        node_sizes.append(max(8, min(30, followers / 10000 + 8)))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='#ffffff'),
        ),
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(color='#cccccc')),
            showlegend=False,
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )
    )

    return fig


def _create_plotly_network_by_threat(
    G: "nx.Graph",
    pos: dict,
    title: str,
) -> go.Figure:
    """Create network visualization colored by threat level.

    Args:
        G: NetworkX graph
        pos: Node positions dict
        title: Chart title

    Returns:
        Plotly figure
    """
    # Threat level color map
    threat_colors = {
        "violence_inciter": "#ff0000",     # Bright red
        "doxxer": "#ff4500",               # Orange red
        "irgc_operative": "#dc143c",       # Crimson
        "state_propagandist": "#b22222",   # Firebrick
        "amplifier_bot": "#cd5c5c",        # Indian red
        "troll": "#f08080",                # Light coral
        "passive_supporter": "#ffa07a",    # Light salmon
        "unknown": "#808080",              # Gray
    }

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        data = G.nodes[node]
        username = data.get("username", str(node))
        threat_level = data.get("threat_level", "unknown")
        bot_score = data.get("bot_score", 0)
        followers = data.get("followers", 0)

        node_text.append(
            f"@{username}<br>Threat: {threat_level}<br>Bot: {bot_score:.2f}<br>Followers: {followers}"
        )
        node_colors.append(threat_colors.get(threat_level, "#808080"))
        node_sizes.append(max(10, min(40, followers / 5000 + 10)))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='#ffffff'),
        ),
    )

    # Create figure
    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(color='#cccccc')),
            showlegend=False,
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )
    )

    # Add legend for threat levels
    for level, color in threat_colors.items():
        if level != "unknown":
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=level.replace("_", " ").title(),
                showlegend=True,
            ))

    fig.update_layout(
        showlegend=True,
        legend=dict(
            font=dict(color='#cccccc'),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#555555',
        ),
    )

    return fig


def create_amplification_graph(
    seed_accounts: list[Account],
    amplifier_accounts: list[Account],
    max_edges: int = 1000,
) -> Optional[go.Figure]:
    """Create a graph showing who amplifies seed accounts.

    Args:
        seed_accounts: Seed regime accounts
        amplifier_accounts: Accounts that retweet/amplify seeds
        max_edges: Maximum number of edges to show

    Returns:
        Plotly figure or None
    """
    if not NETWORKX_AVAILABLE:
        return None

    G = nx.DiGraph()

    # Add seed accounts as hub nodes
    for acc in seed_accounts:
        G.add_node(
            acc.twitter_id,
            username=acc.username,
            is_seed=True,
            followers=acc.followers_count,
        )

    # Add amplifier accounts
    for acc in amplifier_accounts:
        G.add_node(
            acc.twitter_id,
            username=acc.username,
            is_seed=False,
            bot_score=acc.bot_score or 0,
            followers=acc.followers_count,
        )

    if len(G.nodes()) == 0:
        return None

    # Use radial layout with seeds in center
    seed_ids = [a.twitter_id for a in seed_accounts]
    pos = nx.spring_layout(G, k=1, iterations=50, fixed=seed_ids[:3] if len(seed_ids) >= 3 else None)

    # Color nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        data = G.nodes[node]
        username = data.get("username", str(node))
        is_seed = data.get("is_seed", False)
        bot_score = data.get("bot_score", 0)
        followers = data.get("followers", 0)

        node_text.append(f"@{username}<br>{'SEED' if is_seed else 'Amplifier'}<br>Bot: {bot_score:.2f}")
        node_colors.append("#dc143c" if is_seed else "#cd5c5c")
        node_sizes.append(20 if is_seed else 10)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='#ffffff'),
        ),
    )

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            title=dict(text="Amplification Network", font=dict(color='#cccccc')),
            showlegend=False,
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )
    )

    return fig


def is_network_viz_available() -> bool:
    """Check if network visualization is available."""
    return NETWORKX_AVAILABLE
