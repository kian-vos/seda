"""Twitter scraper using Apify's apidojo actors.

Uses two actors for comprehensive scraping:
- apidojo/tweet-scraper: For tweets, search, user timelines
- apidojo/twitter-user-scraper: For followers, following, retweeters
"""

import time
from datetime import datetime
from typing import Optional

from apify_client import ApifyClient

from seda.config import get_settings
from seda.models import Account, ScraperResult, Tweet


class TwitterScraper:
    """Scraper for Twitter/X using Apify's apidojo actors."""

    # Actor IDs
    TWEET_ACTOR = "apidojo/tweet-scraper"
    USER_ACTOR = "apidojo/twitter-user-scraper"

    def __init__(self):
        settings = get_settings()
        if not settings.apify_api_token:
            raise ValueError("APIFY_API_TOKEN not set in environment")
        self.client = ApifyClient(settings.apify_api_token)
        self.rate_limit_delay = settings.scraper_rate_limit_delay
        self.max_tweets = settings.scraper_max_tweets_per_user
        self.max_retweeters = settings.scraper_max_retweeters

    def _run_actor(self, actor_id: str, run_input: dict) -> ScraperResult:
        """Run an Apify actor and return results."""
        try:
            run = self.client.actor(actor_id).call(run_input=run_input)
            items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
            return ScraperResult(success=True, data={"items": items}, items_count=len(items))
        except Exception as e:
            return ScraperResult(success=False, error=str(e))

    def scrape_user_profile(self, username: str) -> tuple[Optional[Account], ScraperResult]:
        """Scrape a user profile.

        Returns:
            Tuple of (Account or None, ScraperResult)
        """
        run_input = {
            "startUrls": [f"https://twitter.com/{username}"],
            "tweetsDesired": 1,
            "profilesDesired": 1,
        }

        result = self._run_actor(self.TWEET_ACTOR, run_input)
        if not result.success or not result.data:
            return None, result

        items = result.data.get("items", [])
        if not items:
            return None, ScraperResult(success=False, error=f"No profile found for {username}")

        # Find user data in items
        for item in items:
            if item.get("author"):
                account = self._parse_user(item["author"])
                if account:
                    time.sleep(self.rate_limit_delay)
                    return account, result

        # Try parsing first item as user
        account = self._parse_user(items[0])
        time.sleep(self.rate_limit_delay)
        return account, result

    def scrape_user_tweets(
        self, username: str, max_items: Optional[int] = None
    ) -> tuple[list[Tweet], ScraperResult]:
        """Scrape tweets from a user.

        Returns:
            Tuple of (list of Tweets, ScraperResult)
        """
        max_items = max_items or self.max_tweets

        run_input = {
            "startUrls": [f"https://twitter.com/{username}"],
            "tweetsDesired": max_items,
        }

        result = self._run_actor(self.TWEET_ACTOR, run_input)
        if not result.success or not result.data:
            return [], result

        items = result.data.get("items", [])
        tweets = []

        for item in items:
            tweet = self._parse_tweet(item)
            if tweet:
                tweets.append(tweet)

        time.sleep(self.rate_limit_delay)
        return tweets, result

    def scrape_user_with_tweets(
        self, username: str, max_tweets: Optional[int] = None
    ) -> tuple[Optional[Account], list[Tweet], ScraperResult]:
        """Scrape user profile and their tweets in one call.

        Returns:
            Tuple of (Account or None, list of Tweets, ScraperResult)
        """
        max_tweets = max_tweets or self.max_tweets

        run_input = {
            "startUrls": [f"https://twitter.com/{username}"],
            "tweetsDesired": max_tweets,
        }

        result = self._run_actor(self.TWEET_ACTOR, run_input)
        if not result.success or not result.data:
            return None, [], result

        items = result.data.get("items", [])
        account = None
        tweets = []

        for item in items:
            # Parse tweet
            tweet = self._parse_tweet(item)
            if tweet:
                tweets.append(tweet)

            # Extract user from author field if not already extracted
            if not account and item.get("author"):
                account = self._parse_user(item["author"])

        time.sleep(self.rate_limit_delay)
        return account, tweets, result

    def scrape_tweet_retweeters(
        self, tweet_id: str, max_items: Optional[int] = None
    ) -> tuple[list[Account], ScraperResult]:
        """Scrape users who retweeted a specific tweet.

        Uses apidojo/twitter-user-scraper with getRetweeters option.

        Returns:
            Tuple of (list of Accounts, ScraperResult)
        """
        max_items = max_items or self.max_retweeters

        # twitter-user-scraper expects tweet URLs for retweeters
        # Using x.com format which works better
        run_input = {
            "startUrls": [f"https://x.com/x/status/{tweet_id}"],
            "getRetweeters": True,
            "maxItems": max_items,
        }

        result = self._run_actor(self.USER_ACTOR, run_input)
        if not result.success or not result.data:
            return [], result

        items = result.data.get("items", [])
        accounts = []

        for item in items:
            account = self._parse_user(item)
            if account:
                accounts.append(account)

        time.sleep(self.rate_limit_delay)
        return accounts, result

    def scrape_user_followers(
        self, username: str, max_items: Optional[int] = None
    ) -> tuple[list[Account], ScraperResult]:
        """Scrape followers of a user.

        Uses apidojo/twitter-user-scraper with getFollowers option.

        Returns:
            Tuple of (list of Accounts, ScraperResult)
        """
        max_items = max_items or 1000

        run_input = {
            "startUrls": [f"https://twitter.com/{username}"],
            "getFollowers": True,
            "maxItems": max_items,
        }

        result = self._run_actor(self.USER_ACTOR, run_input)
        if not result.success or not result.data:
            return [], result

        items = result.data.get("items", [])
        accounts = []

        for item in items:
            account = self._parse_user(item)
            if account:
                accounts.append(account)

        time.sleep(self.rate_limit_delay)
        return accounts, result

    def scrape_user_following(
        self, username: str, max_items: Optional[int] = None
    ) -> tuple[list[Account], ScraperResult]:
        """Scrape accounts that a user follows.

        Uses apidojo/twitter-user-scraper with getFollowing option.

        Returns:
            Tuple of (list of Accounts, ScraperResult)
        """
        max_items = max_items or 1000

        run_input = {
            "startUrls": [f"https://twitter.com/{username}"],
            "getFollowing": True,
            "maxItems": max_items,
        }

        result = self._run_actor(self.USER_ACTOR, run_input)
        if not result.success or not result.data:
            return [], result

        items = result.data.get("items", [])
        accounts = []

        for item in items:
            account = self._parse_user(item)
            if account:
                accounts.append(account)

        time.sleep(self.rate_limit_delay)
        return accounts, result

    def scrape_search(
        self, query: str, max_items: int = 100
    ) -> tuple[list[Tweet], list[Account], ScraperResult]:
        """Search tweets by query.

        Returns:
            Tuple of (list of Tweets, list of Accounts, ScraperResult)
        """
        run_input = {
            "searchTerms": [query],
            "tweetsDesired": max_items,
        }

        result = self._run_actor(self.TWEET_ACTOR, run_input)
        if not result.success or not result.data:
            return [], [], result

        items = result.data.get("items", [])
        tweets = []
        accounts_by_id = {}

        for item in items:
            tweet = self._parse_tweet(item)
            if tweet:
                tweets.append(tweet)

            # Extract author account
            if item.get("author"):
                account = self._parse_user(item["author"])
                if account and account.twitter_id not in accounts_by_id:
                    accounts_by_id[account.twitter_id] = account

        time.sleep(self.rate_limit_delay)
        return tweets, list(accounts_by_id.values()), result

    def _parse_user(self, data: dict) -> Optional[Account]:
        """Parse Apify user data into Account model."""
        if not data:
            return None

        # Handle nested user object
        if "user" in data:
            data = data["user"]

        # Extract Twitter ID (try various field names)
        twitter_id = str(
            data.get("id_str")
            or data.get("id")
            or data.get("rest_id")
            or data.get("userId")
            or data.get("id_str")
            or ""
        )
        if not twitter_id:
            return None

        # Extract username (handle various field names from different actors)
        username = (
            data.get("userName")
            or data.get("username")
            or data.get("screen_name")
            or data.get("handle")
            or data.get("screenName")
            or ""
        )
        if isinstance(username, str):
            username = username.lower().lstrip("@")
        if not username:
            return None

        # Parse created_at
        created_at = None
        created_str = data.get("created_at") or data.get("createdAt")
        if created_str:
            try:
                # Twitter date format: "Wed Oct 10 20:19:24 +0000 2012"
                created_at = datetime.strptime(created_str, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                try:
                    created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

        return Account(
            twitter_id=twitter_id,
            username=username,
            display_name=data.get("name") or data.get("displayName") or data.get("fullName") or "",
            bio=data.get("description") or data.get("bio") or data.get("rawDescription") or "",
            followers_count=data.get("followers") or data.get("followers_count") or data.get("followersCount") or 0,
            following_count=data.get("following") or data.get("friends_count") or data.get("followingCount") or data.get("friendsCount") or 0,
            tweet_count=data.get("statuses_count") or data.get("statusesCount") or data.get("tweetsCount") or data.get("tweets") or 0,
            created_at=created_at,
            profile_image_url=data.get("profilePicture") or data.get("profile_image_url_https") or data.get("profileImageUrl") or data.get("profileImageUrlHttps") or "",
            is_verified=data.get("isBlueVerified") or data.get("verified") or data.get("isVerified") or False,
            location=data.get("location") or "",
        )

    def _parse_tweet(self, data: dict) -> Optional[Tweet]:
        """Parse Apify tweet data into Tweet model."""
        if not data:
            return None

        # Extract tweet ID
        tweet_id = str(
            data.get("id_str")
            or data.get("id")
            or data.get("tweetId")
            or data.get("rest_id")
            or ""
        )
        if not tweet_id:
            return None

        # Extract text
        text = (
            data.get("fullText")
            or data.get("full_text")
            or data.get("text")
            or data.get("content")
            or ""
        )

        # Parse created_at
        created_at = None
        created_str = data.get("created_at") or data.get("createdAt")
        if created_str:
            try:
                created_at = datetime.strptime(created_str, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                try:
                    created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

        # Extract hashtags
        hashtags = []
        entities = data.get("entities", {})
        if "hashtags" in entities:
            hashtags = [h.get("text", "") for h in entities["hashtags"]]
        elif "hashtags" in data:
            hashtags = data["hashtags"] if isinstance(data["hashtags"], list) else []

        # Extract mentions
        mentions = []
        if "user_mentions" in entities:
            mentions = [m.get("screen_name", "") for m in entities["user_mentions"]]
        elif "mentions" in data:
            mentions = data["mentions"] if isinstance(data["mentions"], list) else []

        # Extract URLs
        urls = []
        if "urls" in entities:
            urls = [u.get("expanded_url") or u.get("url", "") for u in entities["urls"]]
        elif "urls" in data:
            urls = data["urls"] if isinstance(data["urls"], list) else []

        # Determine tweet type
        is_retweet = data.get("retweeted_status") is not None or data.get("isRetweet", False)
        is_reply = data.get("in_reply_to_status_id") is not None or data.get("isReply", False)
        is_quote = data.get("is_quote_status", False) or data.get("isQuote", False)

        # Referenced tweet ID
        referenced_id = None
        if is_retweet and data.get("retweeted_status"):
            referenced_id = str(data["retweeted_status"].get("id_str", ""))
        elif is_reply:
            referenced_id = str(data.get("in_reply_to_status_id_str") or data.get("in_reply_to_status_id") or "")
        elif is_quote and data.get("quoted_status"):
            referenced_id = str(data["quoted_status"].get("id_str", ""))

        return Tweet(
            tweet_id=tweet_id,
            account_id=0,  # Will be set when associating with account
            text=text,
            created_at=created_at,
            hashtags=hashtags,
            mentions=mentions,
            urls=urls,
            retweet_count=data.get("retweet_count") or data.get("retweetCount") or data.get("retweets") or 0,
            like_count=data.get("favorite_count") or data.get("likeCount") or data.get("favoriteCount") or data.get("likes") or 0,
            reply_count=data.get("reply_count") or data.get("replyCount") or data.get("replies") or 0,
            is_retweet=is_retweet,
            is_reply=is_reply,
            is_quote=is_quote,
            referenced_tweet_id=referenced_id if referenced_id else None,
            collected_at=datetime.now(),
        )
