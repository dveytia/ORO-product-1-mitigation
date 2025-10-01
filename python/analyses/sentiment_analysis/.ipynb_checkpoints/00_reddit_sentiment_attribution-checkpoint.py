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
import hashlib


## Input data format
"""
[
 {
    'oro':oro,
    'query':query,
    'submission title': submission.title,
    'submission ups':submission.score,
    'submission downs':submission.downs,
    'submission num comments': submission.num_comments,
    'submission url':submission.url,
    'submission id': submission.id,
    'submission date':dateparser.parse(str(submission.created_utc)).date().isoformat(), "2024-01-31"
    'comments': [
        {
            'comment body', 'comment ups', comment downs', 'comment date'
        }
 }
]

"""


## Constant variables

dataFolder = '/homedata/dveytia/Product_1_data'
INPUT_DIR = Path(f'{dataFolder}/data/webscraping_data/reddit_posts')
OUTPUT_FILE = Path(f"{dataFolder}/outputs/sentiment_predictions/reddit_sentiments.jsonl")
MAX_WORKERS = 5


## Define piplines to use
# Topic
topic_path = f"cardiffnlp/tweet-topic-latest-multi"
topic_task = pipeline(task = "text-classification", model = topic_path, tokenizer=topic_path, return_all_scores=True)
# Sentiment
sentiment_path='cardiffnlp/twitter-roberta-base-sentiment-latest'
sentiment_task = pipeline("sentiment-analysis", model=sentiment_path, tokenizer=sentiment_path)

# Load processed ids (submission ids and comment ids) to skip already done
processed_ids = set()
if OUTPUT_FILE.exists():
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed_ids.add(record.get("post_id"))  # common key for submission/comment ids
            except json.JSONDecodeError:
                continue

lock = threading.Lock()

# make a unique comment id
def generate_comment_id(parent_id, comment_body, comment_date):
    base_str = f"{parent_id}_{comment_body[:20]}_{comment_date}"
    return int(hashlib.md5(base_str.encode('utf-8')).hexdigest()[:16], 16)

# Generator to stream submissions and comments one-by-one from Reddit JSON files
def stream_reddit_posts(input_dir, processed_ids):
    files = glob.glob(str(input_dir / "*.txt"))
    # print(f"Found {len(files)} files in {input_dir}")
    
    for file_path in glob.glob(str(input_dir / "*.txt")):
        # print(f"Loading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                submission = json.load(f)
                # print(f"Loaded submission with keys: {list(submission.keys())}")
                sub_id = submission.get("submission id")
                # print(f"Submission ID: {sub_id}")
                oro = submission.get("oro")
                query = submission.get("query")
                if sub_id not in processed_ids:
                    submission["type"] = "submission"
                    # print(f"Yielding submission: {sub_id}")
                    yield submission
                # iterate comments
                comments = submission.get("comments", [])
                for comment in comments:
                    comment_body = comment.get("comment body", "")
                    comment_date = comment.get("comment date", "")
                    comment_id = generate_comment_id(sub_id, comment_body, comment_date)
                    if comment_id not in processed_ids:
                        comment['oro'] = oro
                        comment['query'] = query
                        comment["type"] = "comment"
                        comment["parent_submission_id"] = sub_id
                        comment["comment id"] = comment_id
                        yield comment
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

def process_reddit_item(item):
    try:
        item_type = item.get("type")
        text = None
        id_ = None
        meta = {}

        if item_type == "submission":
            id_ = item.get("submission id")
            text = item.get("submission title", "")
            meta = {
                "up_count": item.get("submission ups"),
                "down_count": item.get("submission downs"),
                "num_comments": item.get("submission num comments"),
                "post_date": item.get("submission date"),
            }

        elif item_type == "comment":
            id_ = item.get("comment id") or item.get("id") or None
            text = item.get("comment body", "")
            meta = {
                "parent_post_id": item.get("parent_submission_id"),
                "up_count": item.get("comment ups"),
                "down_count": item.get("comment downs"),
                "post_date": item.get("comment date"),
            }
            if id_ is None:
                # Skip comments without IDs
                return None

        else:
            # Unknown type - skip
            return None

        # Topic classification
        topic_result = topic_task(text)
        flat_results = topic_result[0]  # flatten first-level list
        label_threshold = 0.5
        labels = {entry["label"] for entry in flat_results if entry["score"] > label_threshold}

        allowed_labels = {"news_&_social_concern", "science_&_technology", "business_&_entrepreneurs"}

        if not (labels & allowed_labels) or "gaming" in labels:
            # Save skipped record
            skipped_record = {
                "oro_type": item.get("oro"),
                "query": item.get("query"),
                "source": "reddit",
                "post_id": id_,
                "post_type": item_type,
                "skipped": True,
                "reason": "irrelevant_topic"
            }
            with lock:
                with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(skipped_record) + "\n")
            return None

        # Sentiment classification
        sentiment_result = sentiment_task(text)
        sentiment_result = sentiment_result[0]

        # Build output record
        output = {
            "oro_type": item.get("oro"),
            "query": item.get("query"),
            "source": "reddit",
            "post_id": id_,
            "post_type": item_type,
            "post_body": text,
            "post_sentiment": sentiment_result.get("label"),
            "sentiment_score": sentiment_result.get("score"),
            **meta
        }

        with lock:
            with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(output) + "\n")

        return output

    except Exception as e:
        print(f"Failed to process {item.get('id') or item.get('submission id')}: {e}")
        return None

def main():
    item_generator = stream_reddit_posts(INPUT_DIR, processed_ids)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_reddit_item, item): item for item in item_generator}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Reddit items"):
            future.result()

if __name__ == "__main__":
    main()