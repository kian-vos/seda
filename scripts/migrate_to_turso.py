#!/usr/bin/env python3
"""Migrate local SQLite database to Turso cloud."""

import os
import sys
import sqlite3

# Load environment
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import libsql_experimental as libsql

def main():
    turso_url = os.getenv('TURSO_DATABASE_URL')
    turso_token = os.getenv('TURSO_AUTH_TOKEN')

    if not turso_url or not turso_token:
        print('Error: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set')
        sys.exit(1)

    # Connect to local
    local = sqlite3.connect('data/seda.db')
    local.row_factory = sqlite3.Row

    # Connect to Turso
    print(f'Connecting to Turso: {turso_url[:50]}...')
    turso = libsql.connect(database=turso_url, auth_token=turso_token)

    # Get counts
    local_accounts = local.execute('SELECT COUNT(*) FROM accounts').fetchone()[0]
    local_tweets = local.execute('SELECT COUNT(*) FROM tweets').fetchone()[0]
    local_seeds = local.execute('SELECT COUNT(*) FROM seed_accounts').fetchone()[0]
    local_clusters = local.execute('SELECT COUNT(*) FROM coordination_clusters').fetchone()[0]

    print(f'\nLocal database:')
    print(f'  Accounts: {local_accounts}')
    print(f'  Tweets: {local_tweets}')
    print(f'  Seeds: {local_seeds}')
    print(f'  Clusters: {local_clusters}')

    # Migrate accounts
    print('\nMigrating accounts...')
    accounts = local.execute('SELECT * FROM accounts').fetchall()
    for i, acc in enumerate(accounts):
        turso.execute('''
            INSERT OR REPLACE INTO accounts (
                twitter_id, username, display_name, bio, followers_count,
                following_count, tweet_count, created_at, profile_image_url,
                is_verified, location, bot_score, regime_score, coordination_score,
                account_type, political_stance, political_taxonomy, is_seed,
                seed_category, first_seen, last_updated, features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            acc['twitter_id'], acc['username'], acc['display_name'], acc['bio'],
            acc['followers_count'], acc['following_count'], acc['tweet_count'],
            acc['created_at'], acc['profile_image_url'], acc['is_verified'],
            acc['location'], acc['bot_score'], acc['regime_score'], acc['coordination_score'],
            acc['account_type'], acc['political_stance'], acc['political_taxonomy'],
            acc['is_seed'], acc['seed_category'], acc['first_seen'], acc['last_updated'],
            acc['features']
        ))
        if (i + 1) % 500 == 0:
            turso.commit()
            print(f'  {i + 1}/{len(accounts)} accounts')
    turso.commit()
    print(f'  Done: {len(accounts)} accounts')

    # Migrate tweets
    print('\nMigrating tweets...')
    tweets = local.execute('SELECT * FROM tweets').fetchall()
    for i, t in enumerate(tweets):
        turso.execute('''
            INSERT OR REPLACE INTO tweets (
                tweet_id, account_id, text, created_at, hashtags, mentions,
                urls, retweet_count, like_count, reply_count, is_retweet,
                is_reply, is_quote, referenced_tweet_id, sentiment,
                regime_alignment, talking_points, collected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            t['tweet_id'], t['account_id'], t['text'], t['created_at'],
            t['hashtags'], t['mentions'], t['urls'], t['retweet_count'],
            t['like_count'], t['reply_count'], t['is_retweet'], t['is_reply'],
            t['is_quote'], t['referenced_tweet_id'], t['sentiment'],
            t['regime_alignment'], t['talking_points'], t['collected_at']
        ))
        if (i + 1) % 500 == 0:
            turso.commit()
            print(f'  {i + 1}/{len(tweets)} tweets')
    turso.commit()
    print(f'  Done: {len(tweets)} tweets')

    # Migrate seeds
    print('\nMigrating seed accounts...')
    seeds = local.execute('SELECT * FROM seed_accounts').fetchall()
    for s in seeds:
        turso.execute('''
            INSERT OR REPLACE INTO seed_accounts (username, category, added_at, notes)
            VALUES (?, ?, ?, ?)
        ''', (s['username'], s['category'], s['added_at'], s['notes']))
    turso.commit()
    print(f'  Done: {len(seeds)} seeds')

    # Migrate clusters
    print('\nMigrating clusters...')
    clusters = local.execute('SELECT * FROM coordination_clusters').fetchall()
    for c in clusters:
        turso.execute('''
            INSERT OR REPLACE INTO coordination_clusters (
                cluster_type, created_at, member_account_ids, evidence,
                confidence_score, description, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            c['cluster_type'], c['created_at'], c['member_account_ids'],
            c['evidence'], c['confidence_score'], c['description'], c['is_active']
        ))
    turso.commit()
    print(f'  Done: {len(clusters)} clusters')

    # Verify
    print('\nVerifying Turso database...')
    t_accounts = turso.execute('SELECT COUNT(*) FROM accounts').fetchone()[0]
    t_tweets = turso.execute('SELECT COUNT(*) FROM tweets').fetchone()[0]
    t_seeds = turso.execute('SELECT COUNT(*) FROM seed_accounts').fetchone()[0]
    t_clusters = turso.execute('SELECT COUNT(*) FROM coordination_clusters').fetchone()[0]

    print(f'\nTurso database:')
    print(f'  Accounts: {t_accounts}')
    print(f'  Tweets: {t_tweets}')
    print(f'  Seeds: {t_seeds}')
    print(f'  Clusters: {t_clusters}')

    local.close()
    turso.close()

    print('\nMigration complete!')

if __name__ == '__main__':
    main()
