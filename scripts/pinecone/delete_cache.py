"""
delete_cache.py
================
Deletes all semantic cache entries from Pinecone namespace 'cache_pharma'.
Run this when you want to force fresh research_agent calls for testing.

Usage:
    cd ~/PycharmProjects/vs-agentcore-multiagent/scripts
    source .env.prod
    python3 delete_cache.py

    # Delete only a specific user's cache:
    python3 delete_cache.py --user-id <user_id>

    # Dry run — show what would be deleted:
    python3 delete_cache.py --dry-run
"""

import argparse
import json
import logging
import os
import sys

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SSM_PREFIX      = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
REGION          = os.environ.get("AWS_REGION", "us-east-1")
CACHE_NAMESPACE = "cache_pharma"   # namespace within clinical-trials-index


def get_pinecone_index():
    sm  = boto3.client("secretsmanager", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)

    secret     = json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/pinecone")["SecretString"])
    index_name = ssm.get_parameter(Name=f"{SSM_PREFIX}/pinecone/clinical_trials_index")["Parameter"]["Value"]

    from pinecone import Pinecone
    pc = Pinecone(api_key=secret["api_key"])
    # Cache lives in the same index as clinical trial chunks, under namespace 'cache_pharma'
    return pc.Index(index_name)


def delete_all_cache(index, dry_run=False):
    log.info(f"Deleting ALL vectors in namespace '{CACHE_NAMESPACE}'...")
    if dry_run:
        stats = index.describe_index_stats()
        ns    = stats.get("namespaces", {}).get(CACHE_NAMESPACE, {})
        count = ns.get("vector_count", 0)
        log.info(f"DRY RUN — would delete {count} vectors from '{CACHE_NAMESPACE}'")
        return count

    index.delete(delete_all=True, namespace=CACHE_NAMESPACE)
    log.info(f"✅ All cache entries deleted from namespace '{CACHE_NAMESPACE}'")
    return -1


def delete_user_cache(index, user_id: str, dry_run=False):
    log.info(f"Deleting cache for user_id='{user_id}'...")
    filter_expr = {"user_id": {"$eq": user_id}}
    if dry_run:
        log.info(f"DRY RUN — would delete vectors matching filter: {filter_expr}")
        return
    index.delete(filter=filter_expr, namespace=CACHE_NAMESPACE)
    log.info(f"✅ Cache deleted for user_id='{user_id}'")


def show_stats(index):
    stats = index.describe_index_stats()
    ns    = stats.get("namespaces", {})
    log.info("Current namespace stats:")
    for name, info in ns.items():
        if "cache" in name:
            log.info(f"  {name}: {info.get('vector_count', 0)} vectors")


def main():
    parser = argparse.ArgumentParser(description="Delete VS AgentCore semantic cache")
    parser.add_argument("--user-id", help="Delete cache for specific user only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--stats",   action="store_true", help="Show cache stats only")
    args = parser.parse_args()

    log.info(f"Connecting to Pinecone  SSM_PREFIX={SSM_PREFIX}  REGION={REGION}")
    index = get_pinecone_index()

    if args.stats:
        show_stats(index)
        return

    show_stats(index)

    if args.user_id:
        delete_user_cache(index, args.user_id, dry_run=args.dry_run)
    else:
        if not args.dry_run:
            confirm = input(f"\nDelete ALL cache entries from '{CACHE_NAMESPACE}'? [y/N]: ")
            if confirm.lower() != "y":
                log.info("Aborted.")
                sys.exit(0)
        delete_all_cache(index, dry_run=args.dry_run)

    show_stats(index)


if __name__ == "__main__":
    main()