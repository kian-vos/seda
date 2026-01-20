"""Streamlit dashboard for SEDA."""

import sys
from pathlib import Path

# Add project root to path for Streamlit Cloud
sys.path.insert(0, str(Path(__file__).parent.parent))

import base64
import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from seda.db import get_db
from seda.models import PoliticalStance, AccountType, CoordinationType


def get_logo_base64():
    """Load logo as base64 for embedding."""
    logo_path = Path(__file__).parent / "assets" / "seda_logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Page config
st.set_page_config(
    page_title="SEDA - Expose the Regime",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark revolutionary theme CSS
st.markdown("""
<style>
    /* Main dark theme */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a0a0a 50%, #0a0a0a 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0505 0%, #0d0d0d 100%);
        border-right: 2px solid #8b0000;
    }

    /* Headers with blood red accent */
    h1 {
        color: #dc143c !important;
        text-shadow: 0 0 10px rgba(220, 20, 60, 0.5);
        font-weight: 800 !important;
        letter-spacing: 2px;
    }

    h2, h3 {
        color: #ff4444 !important;
        border-bottom: 1px solid #8b0000;
        padding-bottom: 10px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 100%);
        border: 1px solid #8b0000;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(139, 0, 0, 0.3);
    }

    [data-testid="stMetricValue"] {
        color: #ff6b6b !important;
        font-weight: bold;
        font-size: 2.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #cccccc !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(139, 0, 0, 0.2) !important;
        border: 1px solid #8b0000 !important;
        color: #ffcccc !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #8b0000;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b0000 0%, #dc143c 100%);
        color: white;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #dc143c 0%, #ff4444 100%);
        box-shadow: 0 0 20px rgba(220, 20, 60, 0.5);
    }

    /* Progress text */
    .mission-text {
        color: #ff6b6b;
        font-style: italic;
        text-align: center;
        padding: 10px;
        border-left: 3px solid #dc143c;
        margin: 20px 0;
        background: rgba(220, 20, 60, 0.1);
    }

    /* Custom divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #8b0000, transparent);
    }

    /* Card container */
    .regime-card {
        background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 100%);
        border: 1px solid #8b0000;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }

    /* Warning badge for regime accounts */
    .regime-badge {
        background: #dc143c;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(139, 0, 0, 0.2) !important;
        border: 1px solid #8b0000 !important;
    }

    /* Expander text contrast fix */
    [data-testid="stExpander"] {
        border: 1px solid #8b0000 !important;
        background-color: rgba(26, 5, 5, 0.8) !important;
    }

    [data-testid="stExpander"] summary {
        color: #ff6b6b !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: #ff9999 !important;
    }

    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        color: #e0e0e0 !important;
    }

    /* General text contrast improvements */
    p, span, div {
        color: #d0d0d0;
    }

    /* Sidebar collapse button - make it more visible */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #dc143c 0%, #8b0000 100%) !important;
        border: 2px solid #ff4444 !important;
        border-radius: 8px !important;
        width: 48px !important;
        height: 48px !important;
        min-width: 48px !important;
        min-height: 48px !important;
    }

    button[data-testid="stSidebarCollapseButton"] svg,
    button[data-testid="collapsedControl"] svg {
        width: 28px !important;
        height: 28px !important;
        stroke: white !important;
        stroke-width: 3px !important;
    }

    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="collapsedControl"]:hover {
        background: linear-gradient(135deg, #ff4444 0%, #dc143c 100%) !important;
        box-shadow: 0 0 20px rgba(220, 20, 60, 0.7) !important;
        transform: scale(1.1) !important;
    }

    /* Mobile: make expand button even bigger */
    @media (max-width: 768px) {
        button[data-testid="collapsedControl"] {
            width: 60px !important;
            height: 60px !important;
            min-width: 60px !important;
            min-height: 60px !important;
            position: fixed !important;
            top: 10px !important;
            left: 10px !important;
            z-index: 9999 !important;
        }

        button[data-testid="collapsedControl"] svg {
            width: 36px !important;
            height: 36px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def init_db():
    return get_db()

db = init_db()


def main():
    """Main dashboard application."""
    # Sidebar with logo
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 0 0 10px 0;">
            <img src="data:image/png;base64,{logo_b64}" style="height: 210px; width: auto;">
            <p style="color: #9CA3AF; font-size: 0.85em; margin: 5px 0 0 0;">State-linked Entity Detection & Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("# SEDA")
        st.sidebar.markdown("**State-linked Entity Detection & Analysis**")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="mission-text">
    Exposing regime propaganda networks.
    For those who fight for freedom.
    </div>
    """, unsafe_allow_html=True)

    # Navigation - check if we need to override from button click
    nav_options = ["Overview", "Accounts", "Coordination", "Network", "Export"]

    # If a button set the page, use that; otherwise use radio selection
    if "navigate_to" in st.session_state:
        page = st.session_state.pop("navigate_to")
    else:
        page = st.sidebar.radio(
            "Navigate",
            nav_options,
        )

    # Show current page indicator if navigated programmatically
    if page != "Overview":
        st.sidebar.markdown(f"**Currently viewing: {page}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Remember:**
    Every account exposed is a victory
    for truth and freedom.
    """)

    if page == "Overview":
        show_home()
    elif page == "Accounts":
        show_accounts()
    elif page == "Coordination":
        show_coordination()
    elif page == "Network":
        show_network()
    elif page == "Export":
        show_export()


def show_home():
    """Home page with overview statistics."""
    st.markdown("# EXPOSE THE REGIME")
    st.markdown("""
    <div class="mission-text">
    Identifying and exposing Iranian regime-affiliated Twitter accounts that spread
    propaganda while innocent people die fighting for freedom.
    </div>
    """, unsafe_allow_html=True)

    # Get stats
    stats = db.get_stats()
    accounts = db.get_all_accounts()

    # Calculate additional stats
    scored = [a for a in accounts if a.bot_score is not None]
    high_bot = [a for a in accounts if a.bot_score and a.bot_score >= 0.7]
    pro_regime = [a for a in accounts if a.political_stance == PoliticalStance.PRO_REGIME]

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accounts Tracked", f"{stats['accounts']:,}")

    with col2:
        st.metric("Pro-Regime Identified", f"{len(pro_regime):,}")

    with col3:
        st.metric("Tweets Collected", f"{stats['tweets']:,}")

    with col4:
        st.metric("Coordination Clusters", f"{stats['clusters']:,}")

    st.markdown("---")

    # Bot detection stats
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Bot Detection")
        if len(scored) > 0:
            bot_data = {
                "Category": ["Suspected Bots (>=0.7)", "Under Review"],
                "Count": [
                    len(high_bot),
                    len(scored) - len(high_bot),
                ],
            }
            fig = px.pie(
                bot_data,
                values="Count",
                names="Category",
                color_discrete_sequence=["#dc143c", "#333333"],
                hole=0.4,
            )
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#cccccc',
                legend=dict(font=dict(color='#cccccc')),
            )
            fig.update_traces(
                textinfo='percent+value',
                textfont_color='white',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Run analysis to detect bots: `python -m scripts.analyze bot`")

    with col2:
        st.markdown("### Stance Classification")
        st.caption("Click a bar to filter accounts by stance")
        if accounts:
            stance_counts = {}
            for acc in accounts:
                stance = acc.political_stance.value
                stance_counts[stance] = stance_counts.get(stance, 0) + 1

            stance_data = {
                "Stance": list(stance_counts.keys()),
                "Count": list(stance_counts.values()),
            }
            fig = px.bar(
                stance_data,
                x="Stance",
                y="Count",
                color="Stance",
                color_discrete_map={
                    "pro_regime": "#dc143c",
                    "anti_regime": "#2ecc71",
                    "neutral": "#555555",
                    "unknown": "#333333",
                },
            )
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#cccccc',
                showlegend=False,
                xaxis=dict(gridcolor='#333'),
                yaxis=dict(gridcolor='#333'),
            )
            selected = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="stance_chart")

            # Handle chart click to navigate to accounts (only for pro_regime)
            if selected and selected.selection and selected.selection.points:
                clicked_stance = selected.selection.points[0].get("x")
                if clicked_stance and clicked_stance != "anti_regime":
                    st.session_state["filter_stance"] = clicked_stance
                    st.session_state["navigate_to"] = "Accounts"
                    st.rerun()
        else:
            st.info("No accounts in database yet.")

    # Quick links to filtered views
    st.markdown("---")
    st.markdown("### Browse Accounts")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(f"Pro-Regime ({len(pro_regime)})", use_container_width=True, type="primary"):
            st.session_state["filter_stance"] = "pro_regime"
            st.session_state["navigate_to"] = "Accounts"
            st.rerun()
    with col2:
        if st.button(f"Suspected Bots ({len(high_bot)})", use_container_width=True):
            st.session_state["filter_min_bot"] = 0.7
            st.session_state["navigate_to"] = "Accounts"
            st.rerun()
    with col3:
        if st.button("View All Accounts", use_container_width=True):
            st.session_state["filter_stance"] = None
            st.session_state["filter_min_bot"] = None
            st.session_state["navigate_to"] = "Accounts"
            st.rerun()


def show_accounts():
    """Accounts search and detail view."""
    st.markdown("# Account Database")

    # Get filter defaults from session state (set from Overview page)
    default_stance = st.session_state.get("filter_stance", "All")
    default_min_bot = st.session_state.get("filter_min_bot", 0.0) or 0.0

    # Clear session state after reading
    if "filter_stance" in st.session_state:
        del st.session_state["filter_stance"]
    if "filter_min_bot" in st.session_state:
        del st.session_state["filter_min_bot"]

    # Map stance value to index for selectbox (exclude anti_regime to protect opposition)
    stance_options = ["All", "pro_regime", "neutral", "unknown"]
    stance_index = stance_options.index(default_stance) if default_stance in stance_options else 0

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        search_query = st.text_input("Search username", "", placeholder="@username")

    with col2:
        stance_filter = st.selectbox(
            "Stance",
            stance_options,
            index=stance_index,
        )

    with col3:
        min_bot = st.slider("Min Bot Score", 0.0, 1.0, default_min_bot, 0.1)

    with col4:
        is_seed = st.checkbox("Verified seeds only")

    # Get filtered accounts
    stance = None if stance_filter == "All" else PoliticalStance(stance_filter)

    accounts = db.get_all_accounts(
        is_seed=is_seed if is_seed else None,
        min_bot_score=min_bot if min_bot > 0 else None,
        stance=stance,
        limit=200,
    )

    # Filter by search query
    if search_query:
        search_lower = search_query.lower().lstrip("@")
        accounts = [a for a in accounts if search_lower in a.username.lower()]

    st.markdown(f"**{len(accounts)} accounts found**")

    if not accounts:
        st.info("No accounts match the filters.")
        return

    # Display accounts table
    data = []
    for acc in accounts:
        data.append({
            "Username": f"https://x.com/{acc.username}",
            "Display Name": acc.display_name or "",
            "Followers": acc.followers_count,
            "Bot Score": acc.bot_score,
            "Stance": acc.political_stance.value,
            "Coord Score": acc.coordination_score,
            "Type": acc.account_type.value,
            "Verified Seed": "YES" if acc.is_seed else "",
        })

    df = pd.DataFrame(data)

    # Make username clickable for detail view
    selected_idx = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Username": st.column_config.LinkColumn(display_text=r"https://x\.com/(.*)")
        }
    )

    # Show detail view if account selected
    if selected_idx and selected_idx.selection.rows:
        idx = selected_idx.selection.rows[0]
        selected_url = df.iloc[idx]["Username"]
        selected_username = selected_url.replace("https://x.com/", "")
        show_account_detail(selected_username)


def show_account_detail(username: str):
    """Show detailed view for a single account."""
    account = db.get_account_by_username(username)
    if not account:
        st.error(f"Account not found: @{username}")
        return

    st.markdown("---")

    # Header with stance indicator
    stance_color = {
        "pro_regime": "#dc143c",
        "anti_regime": "#2ecc71",
        "neutral": "#555555",
        "unknown": "#333333",
    }

    st.markdown(f"""
    ### <a href="https://x.com/{account.username}" target="_blank" style="color: #ff6b6b; text-decoration: none;">@{account.username}</a>
    <span style="background-color: {stance_color.get(account.political_stance.value, '#333')};
                 color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">
    {account.political_stance.value.upper().replace('_', ' ')}
    </span>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Profile**")
        st.write(f"Display Name: {account.display_name or 'N/A'}")
        st.write(f"Bio: {account.bio or 'N/A'}")
        st.write(f"Location: {account.location or 'N/A'}")
        st.write(f"Followers: {account.followers_count:,}")
        st.write(f"Following: {account.following_count:,}")
        st.write(f"Tweets: {account.tweet_count:,}")
        if account.created_at:
            st.write(f"Created: {account.created_at.strftime('%Y-%m-%d')}")
        st.write(f"Account Type: **{account.account_type.value}**")

    with col2:
        st.markdown("**Risk Assessment**")

        # Score gauge charts
        scores = [
            ("Bot Probability", account.bot_score, "#dc143c"),
            ("Coordination Score", account.coordination_score, "#ff8c00"),
        ]

        for name, score, color in scores:
            if score is not None:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": name, "font": {"color": "#cccccc"}},
                    number={"font": {"color": color}},
                    gauge={
                        "axis": {"range": [0, 1], "tickcolor": "#555"},
                        "bar": {"color": color},
                        "bgcolor": "#1a1a1a",
                        "bordercolor": "#333",
                        "steps": [
                            {"range": [0, 0.3], "color": "#1a2f1a"},
                            {"range": [0.3, 0.7], "color": "#2f2f1a"},
                            {"range": [0.7, 1], "color": "#2f1a1a"},
                        ],
                    },
                ))
                fig.update_layout(
                    height=180,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#cccccc',
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Political Taxonomy:** {account.political_taxonomy.value}")

    # Recent tweets
    st.markdown("**Recent Tweets**")
    tweets = db.get_tweets_by_account(account.id, limit=10)
    if tweets:
        for tweet in tweets:
            with st.container():
                st.markdown(f"_{tweet.created_at.strftime('%Y-%m-%d %H:%M') if tweet.created_at else 'Unknown date'}_")
                st.write(tweet.text)
                col1, col2, col3 = st.columns(3)
                col1.write(f"Likes: {tweet.like_count}")
                col2.write(f"Retweets: {tweet.retweet_count}")
                col3.write(f"Replies: {tweet.reply_count}")
                st.markdown("---")
    else:
        st.info("No tweets collected for this account.")


def show_coordination():
    """Coordination clusters view."""
    st.markdown("# Coordination Detection")
    st.markdown("""
    <div class="mission-text">
    Detecting coordinated inauthentic behavior - when multiple accounts
    work together to amplify regime propaganda.
    </div>
    """, unsafe_allow_html=True)

    # Get clusters
    clusters = db.get_clusters()

    if not clusters:
        st.warning("""
        **No coordination clusters detected yet.**

        Coordination detection requires:
        1. **More accounts** - especially amplifier accounts (not just seeds)
        2. **Run network expansion** to find retweeters
        3. **Run coordination analysis**

        ```bash
        # Find retweeters of regime content
        python -m scripts.collect expand --depth 1 --retweeters 100

        # Run coordination detection
        python -m scripts.analyze coordination
        ```
        """)
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Clusters", len(clusters))

    with col2:
        high_conf = sum(1 for c in clusters if c.confidence_score >= 0.7)
        st.metric("High Confidence", high_conf)

    with col3:
        accounts_involved = set()
        for c in clusters:
            accounts_involved.update(c.member_account_ids)
        st.metric("Accounts Involved", len(accounts_involved))

    st.markdown("---")

    # Cluster type filter
    cluster_type = st.selectbox(
        "Filter by type",
        ["All"] + [ct.value for ct in CoordinationType],
    )

    filtered_clusters = clusters
    if cluster_type != "All":
        filtered_clusters = [c for c in clusters if c.cluster_type.value == cluster_type]

    # Display clusters
    for cluster in filtered_clusters:
        with st.expander(
            f"{cluster.cluster_type.value.upper()} - {len(cluster.member_account_ids)} accounts (confidence: {cluster.confidence_score:.2f})"
        ):
            st.write(cluster.description)

            # Show member accounts
            st.markdown("**Member Accounts:**")
            members = []
            for aid in cluster.member_account_ids[:20]:  # Limit display
                acc = db.get_account(aid)
                if acc:
                    members.append({
                        "Username": f"https://x.com/{acc.username}",
                        "Bot Score": f"{acc.bot_score:.2f}" if acc.bot_score else "N/A",
                        "Stance": acc.political_stance.value,
                    })

            if members:
                st.dataframe(pd.DataFrame(members), use_container_width=True, column_config={
                    "Username": st.column_config.LinkColumn(display_text=r"https://x\.com/(.*)")
                })

            # Show evidence
            st.markdown("**Evidence:**")
            st.json(cluster.evidence)


def show_collection_guide():
    """Guide for collecting more accounts."""
    st.markdown("# Expand the Network")
    st.markdown("""
    <div class="mission-text">
    To identify thousands of pro-regime accounts, you need to expand beyond seeds
    and find the hidden amplifier networks.
    </div>
    """, unsafe_allow_html=True)

    stats = db.get_stats()

    st.markdown(f"""
    ### Current Status
    - **{stats['accounts']} accounts** collected
    - **{stats['tweets']} tweets** analyzed
    - **{stats['seeds']} seed accounts** configured
    """)

    st.markdown("---")

    st.markdown("""
    ### Strategy 1: Follower Collection (Recommended)

    Find accounts that follow regime accounts - these are potential amplifiers.

    ```bash
    # Get followers of a specific account
    python -m scripts.collect followers PressTV --max 500

    # Get followers from ALL seed accounts
    python -m scripts.collect followers --all-seeds --max 200
    ```

    **Expected yield:** 100-500 new accounts per seed
    """)

    st.markdown("---")

    st.markdown("""
    ### Strategy 2: Network Expansion (Retweeters)

    Find accounts that retweet regime content - these are active amplifiers.

    ```bash
    # Expand network by finding retweeters of seed tweets
    python -m scripts.collect expand --depth 2 --tweets 20 --retweeters 200
    ```

    **Parameters:**
    - `--depth 2`: Go 2 levels deep (retweeters of retweeters)
    - `--tweets 20`: Check 20 tweets per seed account
    - `--retweeters 200`: Collect up to 200 retweeters per tweet

    **Expected yield:** 1,000-5,000 new accounts per run
    """)

    st.markdown("---")

    st.markdown("""
    ### Strategy 3: Hashtag/Keyword Search

    Search for tweets using pro-regime hashtags and keywords.

    ```bash
    # Search for pro-regime hashtags (Persian)
    python -m scripts.collect search "#محور_مقاومت" --max 500
    python -m scripts.collect search "#استکبار" --max 500
    python -m scripts.collect search "رهبر معظم" --max 500

    # Search English regime accounts
    python -m scripts.collect search "axis of resistance" --max 500
    python -m scripts.collect search "from:PressTV" --max 200
    ```

    **Expected yield:** 200-500 unique accounts per search
    """)

    st.markdown("---")

    st.markdown("""
    ### After Collection: Run Analysis

    ```bash
    # Run full analysis pipeline
    python -m scripts.analyze all --no-llm

    # Or run with Claude LLM for better stance detection (costs ~$0.01/account)
    python -m scripts.analyze all

    # Generate report
    python -m scripts.analyze report
    ```
    """)

    st.markdown("---")

    st.markdown("""
    ### Scaling Tips

    1. **Apify Quota**: Free tier = 10K results/month. Upgrade for more (~$49/month for 100K)
    2. **Run overnight**: Large expansions can take hours
    3. **Incremental**: Run expand multiple times to grow the network
    4. **Focus on amplifiers**: They're more likely to be coordinated than official accounts
    """)

    # Quick action buttons
    st.markdown("---")
    st.markdown("### Quick Commands")

    col1, col2 = st.columns(2)

    with col1:
        st.code("python -m scripts.collect followers --all-seeds --max 200", language="bash")
        st.caption("Collect followers from all seed accounts")

    with col2:
        st.code("python -m scripts.analyze all --no-llm", language="bash")
        st.caption("Run full analysis after collection")


def show_network():
    """Network visualization."""
    st.markdown("# Network Visualization")

    # Simple network stats
    accounts = db.get_all_accounts(is_seed=True)

    if accounts:
        st.markdown("### Seed Account Network")

        # Create simple tree visualization using plotly
        seed_data = []
        for acc in accounts:
            seed_data.append({
                "Username": f"@{acc.username}",
                "Category": acc.seed_category.value if acc.seed_category else "other",
                "Followers": acc.followers_count,
            })

        df = pd.DataFrame(seed_data)

        fig = px.treemap(
            df,
            path=["Category", "Username"],
            values="Followers",
            color="Followers",
            color_continuous_scale=["#1a0505", "#8b0000", "#dc143c", "#ff4444"],
        )
        fig.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#cccccc',
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No seed accounts collected yet.")

    # Coordination network placeholder
    st.markdown("### Coordination Network")
    clusters = db.get_clusters()

    if clusters:
        # Show cluster sizes
        cluster_sizes = [
            {
                "Type": c.cluster_type.value,
                "Members": len(c.member_account_ids),
                "Confidence": c.confidence_score,
            }
            for c in clusters
        ]
        df = pd.DataFrame(cluster_sizes)

        fig = px.scatter(
            df,
            x="Members",
            y="Confidence",
            color="Type",
            size="Members",
            hover_data=["Type"],
            color_discrete_sequence=["#dc143c", "#ff8c00", "#ffd700"],
        )
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cccccc',
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333'),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No coordination clusters detected. Run network expansion first.")


def show_export():
    """Export data view."""
    st.markdown("# Export Data")
    st.markdown("Download identified accounts for further analysis or reporting.")

    # Filters
    st.markdown("### Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        stance_filter = st.selectbox(
            "Stance",
            ["All", "pro_regime", "neutral", "unknown"],
        )

    with col2:
        min_bot = st.slider("Minimum Bot Score", 0.0, 1.0, 0.0, 0.1)

    with col3:
        format_option = st.radio("Format", ["CSV", "JSON"])

    # Get filtered data
    stance = None if stance_filter == "All" else PoliticalStance(stance_filter)

    accounts = db.get_all_accounts(
        min_bot_score=min_bot if min_bot > 0 else None,
        stance=stance,
    )

    st.markdown(f"**{len(accounts)} accounts match filters**")

    if accounts:
        # Prepare data
        data = []
        for acc in accounts:
            data.append({
                "username": acc.username,
                "display_name": acc.display_name,
                "twitter_id": acc.twitter_id,
                "followers": acc.followers_count,
                "following": acc.following_count,
                "tweets": acc.tweet_count,
                "bot_score": acc.bot_score,
                "regime_score": acc.regime_score,
                "coordination_score": acc.coordination_score,
                "account_type": acc.account_type.value,
                "political_stance": acc.political_stance.value,
                "is_seed": acc.is_seed,
            })

        if format_option == "CSV":
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"seda_regime_accounts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"seda_regime_accounts_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )

        # Preview
        st.markdown("### Preview")
        st.dataframe(pd.DataFrame(data).head(20), use_container_width=True)


if __name__ == "__main__":
    main()
