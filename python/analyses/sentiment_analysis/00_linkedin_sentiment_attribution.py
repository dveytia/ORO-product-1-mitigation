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
from collections import defaultdict

## Constant variables

dataFolder = '/homedata/dveytia/Product_1_data'
INPUT_DIR = Path(f'{dataFolder}/data/webscraping_data/linkedin_posts')
PREPROCESSED_DIR = Path(f'{dataFolder}/data/webscraping_data/cleaned_linkedin_posts')  # new
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = Path(f"{dataFolder}/outputs/sentiment_predictions/linkedin_sentiments.jsonl")
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
{
    'oro':oro,
    'query':query,
    'text': text,
    'author': author,
    'date': date,
    'reposts': reposts,
    'impressions': impressions
}

"""


# File to save results


# Load processed post_ids from the output file
processed_post_ids = set()
if OUTPUT_FILE.exists():
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed_post_ids.add(record.get("post_id"))
            except json.JSONDecodeError:
                continue  # skip corrupt lines

# Thread-safe writing lock
lock = threading.Lock()


# Function to pre-process posts - deduplicate, change key name from oro to oro_type, add post id
def preprocess_linkedin_posts(input_dir, output_dir):
    post_buckets = defaultdict(dict)  # {oro_type: {text_hash: post}}
    global_post_id = 0
    for file_path in glob.glob(str(input_dir / "*_linkedin.json")):
        filename = Path(file_path).stem
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        oro_type = "_".join(parts[:-2])  # take everything except last two

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                posts = json.load(f)
                for post in posts:
                    text = post.get("text")
                    if isinstance(text, list):
                        text = text[0].strip()
                    elif isinstance(text, str):
                        text = text.strip()
                    else:
                        text = None
                    if not text:
                        continue
                    text_key = hash(text)
                    # Deduplicate by text hash
                    if text_key not in post_buckets[oro_type]:
                        post["oro_type"] = post.get('oro')
                        post.pop("oro", None)  # remove original oro
                        post["post_id"] = "linkedin_" + str(global_post_id)
                        global_post_id += 1
                        post_buckets[oro_type][text_key] = post
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    # Save per oro_type
    for oro_type, posts_dict in post_buckets.items():
        out_path = output_dir / f"{oro_type}_linkedin_deduped.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(list(posts_dict.values()), f, indent=2)

    print(f"Preprocessing complete. Saved deduplicated files to {output_dir}")


# Generator to stream posts from files without loading all at once
def stream_posts(input_dir, processed_post_ids):
    for file_path in glob.glob(str(input_dir / "*_linkedin_deduped.json")):
        
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                posts = json.load(f)
                for post in posts:
                    if post.get("post_id") not in processed_post_ids:
                        yield post
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

# Function to process each post
def process_post(post):
    try:
        post_id = post.get("post_id")
        text = post.get("text")
        if isinstance(text, list):
            text = text[0].strip()
        elif isinstance(text, str):
            text = text.strip()
        topic_result = topic_task(text)
        flat_results = topic_result[0]  # assume single inner list
        label_threshold = 0.5
        labels = {item["label"] for item in flat_results if item["score"] > label_threshold}
        allowed_labels = {"news_&_social_concern", "science_&_technology", "business_&_entrepreneurs"}

        if not (labels & allowed_labels) or "gaming" in labels:
            skipped_entry = {
                "post_id": post_id,
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
            "oro_type": post.get("oro_type"),
            "query": post.get("query"),
            "source": "linkedin",
            "post_type": "linkedin post",
            "post_id": post_id,
            "post_body": text,
            "post_date": post.get("date"),
            'repost_count': post.get("reposts",0),
            "up_count": post.get("impressions", 0),
            "post_sentiment": sentiment_result.get("label"),
            "sentiment_score": sentiment_result.get("score"),
        }

        with lock:
            with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(output) + "\n")

        return output

    except Exception as e:
        print(f"Failed to process post {post.get('post_id')}: {e}")
        return None

# Stream and process with parallel workers
def main():
    # post_generator = stream_posts(INPUT_DIR, processed_post_ids)
    post_generator = stream_posts(PREPROCESSED_DIR, processed_post_ids)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_post, post): post for post in post_generator}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing posts"):
            future.result()

if __name__ == "__main__":
    # preprocess_linkedin_posts(INPUT_DIR, PREPROCESSED_DIR)
    main()






