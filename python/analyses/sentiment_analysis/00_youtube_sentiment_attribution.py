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
from datetime import datetime
from itertools import islice
import hashlib


## Input data format
"""
[
    {'cid': 'UgzAyTlpBSquJf9gbvV4AaABAg',
    'text': '1:10 The explanation is not really correct. Thepart of the forces on the earth moon line are not really significant. What is significant though are all forces added up that we get applying the same principle all along the earths surface. While the forces are still incredibly small, they act on a huge amount of surface area that will push the water towards the earth moon line.\nWho is cleaning the blades from barnacles etc?',
    'time': 'il y a 3 ans (modifié)',
    'author': '@fg786',
    'channel': 'UCt7iNizveIAPyvvu75X9I1A',
    'votes': '0',
    'replies': '',
    'photo': 'https://yt3.ggpht.com/ytc/AIdro_m2Tjgd3zz6y9OZf4g2pWV2lZJwFVfA1Bey-Lq6Toc=s88-c-k-c0x00ffffff-no-rj',
    'heart': False,
    'reply': False,
    'time_parsed': 1654871700.330307,
    'oro_type': "CCS",
    'video_id': "_-0f_GwQkj0"},
]

"""


## Constant variables

dataFolder = '/homedata/dveytia/Product_1_data'
INPUT_DIR = Path(f'{dataFolder}/data/webscraping_data/youtube_comments')
OUTPUT_FILE = Path(f"{dataFolder}/outputs/sentiment_predictions/youtube_sentiments.jsonl")
MAX_WORKERS = 10
reference_date = datetime(2025, 6, 11) # the date the youtube results were downloaded

# # for testing
# file_path = Path(f'{dataFolder}/data/webscraping_data/youtube_comments/CCS__GlaTsm5Qj0.txt')

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

# Format relative time metadata to date
def format_date(time, reference_date):
    try:
        parsed = parse_date(time, languages=["fr"], settings={"RELATIVE_BASE": reference_date}).date().isoformat()
        if parsed:
            return parsed
        else:
            return None
    except Exception as e: 
        return None
    

# Generator to stream submissions and comments one-by-one from youtube JSON files
def stream_youtube_posts(input_dir, processed_ids):
    files = glob.glob(str(input_dir / "*.txt"))
    # print(f"Found {len(files)} files in {input_dir}")
    
    for file_path in glob.glob(str(input_dir / "*.txt")):
        # print(f"Loading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                posts = json.load(f)
                for post in posts:
                    if post.get("cid") not in processed_ids:
                        post["type"] = "youtube comment"
                        yield post
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

                
def process_youtube_item(item):
    try:
        text = item.get("text", "")
        topic_result = topic_task(text)
        flat_results = topic_result[0]  # assume single inner list
        label_threshold = 0.5
        labels = {item["label"] for item in flat_results if item["score"] > label_threshold}
        allowed_labels = {"news_&_social_concern", "science_&_technology", "business_&_entrepreneurs"}
        
        if not (labels & allowed_labels) or "gaming" in labels:
            # Save skipped record
            skipped_record = {
                "oro_type": item.get("oro"),
                "source": "youtube",
                "post_id": item.get("cid"),
                "post_type": item.get("type"),
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
            "oro_type": item.get("oro_type"),
            "source": "youtube",
            "post_id": item.get("cid"),
            "post_type": item.get("type"),
            "youtube_video_id": item.get('video_id'),
            "post_body": text,
            "post_date": format_date(item.get("time"), reference_date) if item.get("time") else None,
            "up_count": item.get("votes", 0),
            "post_sentiment": sentiment_result.get("label"),
            "sentiment_score": sentiment_result.get("score")
        }

        with lock:
            with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(output) + "\n")

        return output

    except Exception as e:
        print(f"Failed to process {item.get('id') or item.get('cid')}: {e}")
        return None

def main():
    item_generator = stream_youtube_posts(INPUT_DIR, processed_ids)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_youtube_item, item): item for item in item_generator}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing youtube items"):
            future.result()




if __name__ == "__main__":
    main()