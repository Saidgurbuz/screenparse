"""
Dynamic URL scraper - discover URLs from curated sources.
Scrapes popular URL aggregators to find trending/popular sites.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Set
import time
import csv


def scrape_github_trending(language: str = None, num_pages: int = 5) -> List[str]:
    """Scrape GitHub trending repositories."""
    urls = []
    base = "https://github.com/trending"

    for page in range(1, num_pages + 1):
        try:
            url = f"{base}/{language}" if language else base
            url += f"?since=daily&page={page}"

            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Find repo links
            repos = soup.select("h2.h3 a")
            for repo in repos:
                href = repo.get("href", "")
                if href:
                    full_url = f"https://github.com{href}"
                    urls.append(full_url)

            time.sleep(1)  # Be respectful
        except Exception as e:
            print(f"Error scraping GitHub page {page}: {e}")

    return urls


def scrape_hacker_news_top(num_stories: int = 100) -> List[str]:
    """Scrape top stories from Hacker News."""
    urls = []

    try:
        # Use HN API
        response = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10
        )
        story_ids = response.json()[:num_stories]

        for story_id in story_ids:
            try:
                story_response = requests.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                    timeout=10,
                )
                story = story_response.json()

                if "url" in story:
                    urls.append(story["url"])

                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                continue
    except Exception as e:
        print(f"Error scraping Hacker News: {e}")

    return urls


def scrape_product_hunt(num_pages: int = 5) -> List[str]:
    """Scrape Product Hunt to find new products/websites."""
    # Note: Product Hunt requires authentication for API
    # This is a simplified scraper - consider using their official API

    urls = [
        "https://www.producthunt.com",
        # Add specific product pages as needed
    ]

    return urls


def scrape_alexa_top(num_sites: int = 500) -> List[str]:
    """
    Get top websites from various top site lists.
    Note: Alexa rankings are deprecated, using alternatives.
    """
    # Tranco list (replacement for Alexa)
    urls = []

    try:
        # Download Tranco top sites
        response = requests.get("https://tranco-list.eu/top-1m.csv.zip", timeout=30)

        # This returns a zip file - would need to extract
        # For now, use a pre-curated list
        print("Note: Automated Tranco scraping requires zip extraction")

    except Exception as e:
        print(f"Error fetching top sites: {e}")

    return urls


def scrape_awesome_lists() -> List[str]:
    """Scrape URLs from GitHub Awesome lists."""
    awesome_lists = [
        "https://github.com/sindresorhus/awesome",
        "https://github.com/bayandin/awesome-awesomeness",
        "https://github.com/awesome-selfhosted/awesome-selfhosted",
    ]

    urls = []

    for list_url in awesome_lists:
        try:
            response = requests.get(list_url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all links in the README
            article = soup.find("article")
            if article:
                links = article.find_all("a", href=True)
                for link in links:
                    href = link["href"]
                    if href.startswith("http") and "github.com" not in href:
                        urls.append(href)

            time.sleep(2)
        except Exception as e:
            print(f"Error scraping {list_url}: {e}")

    return urls


def combine_and_deduplicate(*url_lists: List[str]) -> List[str]:
    """Combine multiple URL lists and remove duplicates."""
    all_urls: Set[str] = set()

    for url_list in url_lists:
        all_urls.update(url_list)

    return sorted(list(all_urls))


def main():
    """Scrape URLs from multiple sources."""
    print("🕷️  Scraping URLs from curated sources...\n")

    all_urls = []

    # 1. GitHub trending
    print("📊 Scraping GitHub trending...")
    github_urls = scrape_github_trending(num_pages=3)
    all_urls.extend(github_urls)
    print(f"   Found {len(github_urls)} GitHub URLs")

    # 2. Hacker News
    print("\n📰 Scraping Hacker News top stories...")
    hn_urls = scrape_hacker_news_top(num_stories=100)
    all_urls.extend(hn_urls)
    print(f"   Found {len(hn_urls)} HN URLs")

    # 3. Awesome lists
    print("\n⭐ Scraping Awesome lists...")
    awesome_urls = scrape_awesome_lists()
    all_urls.extend(awesome_urls)
    print(f"   Found {len(awesome_urls)} Awesome list URLs")

    # Deduplicate
    unique_urls = combine_and_deduplicate(all_urls)

    # Save
    output = "urls_scraped.csv"
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        for url in unique_urls:
            writer.writerow([url])

    print(f"\n✅ Saved {len(unique_urls)} unique URLs to {output}")


if __name__ == "__main__":
    main()
