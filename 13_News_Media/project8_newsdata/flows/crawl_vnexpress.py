# flows/crawl_vnexpress.py
from prefect import flow, task
import requests
from bs4 import BeautifulSoup
import yaml
import json
from pathlib import Path
from datetime import datetime
from scripts.extract_article import extract_article_generic

DISASTER_KEYWORDS = [
    "bão", "áp thấp", "lũ", "lụt",
    "sạt lở", "động đất", "nắng nóng",
    "hạn hán", "mưa lớn"
]

@task(retries=3, retry_delay_seconds=10)
def get_article_urls(category_url, limit=20):
    res = requests.get(category_url, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    urls = []
    for a in soup.select("h3.title-news a"):
        url = a.get("href")
        if url and url.startswith("https://vnexpress.net"):
            urls.append(url)
    return urls[:limit]

@task(retries=2)
def extract_article_task(url):
    try:
        return extract_article_generic(url)
    except Exception as e:
        return None

@task
def filter_disaster_articles(articles):
    results = []
    for a in articles:
        if not a:
            continue
        text = (a["title"] + " " + a["text"]).lower()
        if any(k in text for k in DISASTER_KEYWORDS):
            results.append(a)
    return results

@flow(name="crawl-vnexpress-disaster")
def crawl_vnexpress_disaster():
    with open("config/categories.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_articles = []

    for cat_url in cfg["vnexpress"]["disaster_categories"]:
        urls = get_article_urls(cat_url)
        articles = extract_article_task.map(urls)
        disaster_articles = filter_disaster_articles(articles)
        all_articles.extend(disaster_articles)

    # Save batch theo ngày
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path("data/raw")
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / f"vnexpress_{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    return len(all_articles)

if __name__ == "__main__":
    crawl_vnexpress_disaster()