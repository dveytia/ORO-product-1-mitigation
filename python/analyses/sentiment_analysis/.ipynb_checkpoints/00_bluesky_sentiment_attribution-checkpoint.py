from transformers import pipeline
import os
import json
import pandas
import glob

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import datasets
from datasets import Dataset, DatasetDict
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from dateparser import parse as parse_date
from itertools import islice

## Constant variables

dataFolder = '/homedata/dveytia/Product_1_data'
INPUT_DIR = Path(f'{dataFolder}/data/webscraping_data/bluesky_posts')
OUTPUT_FILE = Path(f"{dataFolder}/outputs/sentiment_predictions/bluesky_sentiments.jsonl")
MAX_WORKERS = 4


## Define piplines to use
# Topic
topic_path = f"cardiffnlp/tweet-topic-latest-multi"
topic_task = pipeline(task = "text-classification", model = topic_path, tokenizer=topic_path, return_all_scores=True)
# Sentiment
sentiment_path='cardiffnlp/twitter-roberta-base-sentiment-latest'
sentiment_task = pipeline("sentiment-analysis", model=sentiment_path, tokenizer=sentiment_path)



"""
Example record format:
[
 {
    "keyword": "seaweed CDR",
    "uri": "at://did:plc:77lswp42lgjyw36ozuo7kt7e/app.bsky.feed.post/3lqrv5nulu22x",
    "cid": "bafyreihwhqivyzxdf6ztms7rddxhvhwrjc2ttlwyyptskq75mhl4jhzxrm",
    "text": "@manonberger.bsky.social showed at #OOSC that if iron limitation is considered, there’s nowhere in the ocean where it’s suitable to grow kelp for CO₂ removal (CDR). Stay tuned for the paper.",
    "author": "davidho.bsky.social",
    "display_name": "David Ho",
    "created_at": "2025-06-04T13:14:34.610Z",
    "like_count": 48,
    "repost_count": 14
  }
]

"""


# File to save results


# Load processed CIDs from the output file
processed_cids = set()
if OUTPUT_FILE.exists():
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed_cids.add(record.get("post_id"))
            except json.JSONDecodeError:
                continue  # skip corrupt lines

# Thread-safe writing lock
lock = threading.Lock()

# Generator to stream posts from files without loading all at once
def stream_posts(input_dir, processed_cids):
    for file_path in glob.glob(str(input_dir / "*_bluesky.json")):
        oro_type = Path(file_path).stem.split("_")[0]
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                posts = json.load(f)
                for post in posts:
                    if post.get("cid") not in processed_cids:
                        post["oro_type"] = oro_type
                        yield post
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

# Function to process each post
def process_post(post):
    try:
        cid = post.get("cid")
        text = post.get("text", "")
        topic_result = topic_task(text)
        flat_results = topic_result[0]  # assume single inner list
        label_threshold = 0.5
        labels = {item["label"] for item in flat_results if item["score"] > label_threshold}
        allowed_labels = {"news_&_social_concern", "science_&_technology", "business_&_entrepreneurs"}

        if not (labels & allowed_labels) or "gaming" in labels:
            skipped_entry = {
                "post_id": cid,
                "skipped": True,
                "reason": "irrelevant_topic"
            }
            with lock:
                with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(skipped_entry) + "\n")
            return None

        sentiment_result = sentiment_task(text)
        sentiment_result = sentiment_result[0]
        output = {
            "oro_type": post["oro_type"],
            "query": post.get("keyword"),
            "source": "bluesky",
            "post_id": cid,
            "post_body": text,
            "post_date": parse_date(post.get("created_at")).date().isoformat() if post.get("created_at") else None,
            "up_count": post.get("like_count", 0),
            "repost_count": post.get("repost_count", 0),
            "post_sentiment": sentiment_result.get("label"),
            "sentiment_score": sentiment_result.get("score"),
        }

        with lock:
            with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(output) + "\n")

        return output

    except Exception as e:
        print(f"Failed to process post {post.get('cid')}: {e}")
        return None

# Stream and process with parallel workers
def main():
    post_generator = stream_posts(INPUT_DIR, processed_cids)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_post, post): post for post in post_generator}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing posts"):
            future.result()

if __name__ == "__main__":
    main()






