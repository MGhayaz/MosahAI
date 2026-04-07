import requests
import random
import time
import logging
import urllib.parse
from typing import List
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BAD_DOMAINS = [
    # Google junk
    "google.com",
    "youtube.com",
    "accounts.google",
    "maps.google",
    "policies.google",

    # 🚫 BLOCK THESE (IMPORTANT)
    "zhihu.com",
    "zhidao.baidu.com",
    "baidu.com",
    "stackoverflow.com",
    "stackexchange.com",
    "superuser.com",
    "serverfault.com",
    "quora.com",
    "reddit.com",
    "brainly",
    "chegg.com"
]


def fetch_html(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        return response
    except requests.exceptions.RequestException:    
        logger.error(f"Failed to fetch {url}")
        return None


def parse_results(response) -> List[str]:
    if response is None:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for a_tag in soup.find_all("a"):
        href = a_tag.get("href")

        if not href:
            continue

        if "/url?q=" in href:
            try:
                parsed = urllib.parse.urlparse(href)
                params = urllib.parse.parse_qs(parsed.query)

                if "q" in params:
                    url = params["q"][0]
                    links.append(url)

            except Exception:
                continue

        if len(links) >= 5:
            break

    return links


def clean_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned = []

    for url in urls:
        if not any(bad in url.lower() for bad in BAD_DOMAINS):
            if url not in seen:
                seen.add(url)
                cleaned.append(url)

    return cleaned


def search_google(query: str) -> List[str]:
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://www.google.com/search?q={encoded_query}"

    attempt = 0
    max_attempts = 3

    while attempt < max_attempts:
        response = fetch_html(search_url)

        if response and response.status_code == 200:
            raw_urls = parse_results(response)
            cleaned_urls = clean_urls(raw_urls)

            if cleaned_urls:
                logger.info(f"Query: {query} | URLs found: {len(cleaned_urls)}")
                return cleaned_urls

        attempt += 1
        sleep_time = random.uniform(1.5, 3.5)
        time.sleep(sleep_time)

    logger.warning(f"Failed to fetch results for query: {query}")
    return []