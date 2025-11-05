"""
Automatic URL generator for diverse web screenshot dataset.
Generates thousands of URLs across different categories and types.
"""

import csv
import random
from typing import List, Set
from pathlib import Path


# ============================================================================
# CURATED URL LISTS - High-quality, diverse websites
# ============================================================================

# Top websites by category for maximum diversity
SEED_URLS = {
    # Search engines & portals
    "search_portals": [
        "https://www.google.com",
        "https://www.bing.com",
        "https://search.yahoo.com",
        "https://duckduckgo.com",
        "https://www.baidu.com",
        "https://yandex.com",
    ],
    # Social media & communities
    "social_media": [
        "https://twitter.com",
        "https://www.facebook.com",
        "https://www.instagram.com",
        "https://www.linkedin.com",
        "https://www.reddit.com",
        "https://www.pinterest.com",
        "https://www.tiktok.com",
        "https://discord.com",
        "https://www.snapchat.com",
        "https://www.tumblr.com",
        "https://mastodon.social",
    ],
    # News & media
    "news_media": [
        "https://www.nytimes.com",
        "https://www.bbc.com",
        "https://www.cnn.com",
        "https://www.theguardian.com",
        "https://www.reuters.com",
        "https://www.wsj.com",
        "https://www.washingtonpost.com",
        "https://www.bloomberg.com",
        "https://www.forbes.com",
        "https://techcrunch.com",
        "https://www.theverge.com",
        "https://arstechnica.com",
        "https://www.wired.com",
        "https://www.buzzfeed.com",
        "https://medium.com",
    ],
    # E-commerce & shopping
    "ecommerce": [
        "https://www.amazon.com",
        "https://www.ebay.com",
        "https://www.aliexpress.com",
        "https://www.etsy.com",
        "https://www.walmart.com",
        "https://www.target.com",
        "https://www.bestbuy.com",
        "https://www.shopify.com",
        "https://www.wayfair.com",
        "https://www.zappos.com",
        "https://www.asos.com",
        "https://www.nike.com",
        "https://www.adidas.com",
    ],
    # Technology & development
    "tech_dev": [
        "https://github.com",
        "https://stackoverflow.com",
        "https://gitlab.com",
        "https://bitbucket.org",
        "https://www.npmjs.com",
        "https://pypi.org",
        "https://hub.docker.com",
        "https://www.kaggle.com",
        "https://leetcode.com",
        "https://www.hackerrank.com",
        "https://codepen.io",
        "https://replit.com",
        "https://glitch.com",
    ],
    # Documentation & wikis
    "documentation": [
        "https://en.wikipedia.org",
        "https://developer.mozilla.org",
        "https://docs.python.org",
        "https://nodejs.org/docs",
        "https://react.dev",
        "https://vuejs.org",
        "https://angular.io",
        "https://www.tensorflow.org",
        "https://pytorch.org",
        "https://kubernetes.io",
        "https://docs.aws.amazon.com",
    ],
    # Video & streaming
    "video_streaming": [
        "https://www.youtube.com",
        "https://www.netflix.com",
        "https://www.twitch.tv",
        "https://vimeo.com",
        "https://www.hulu.com",
        "https://www.disneyplus.com",
        "https://www.crunchyroll.com",
    ],
    # Music & audio
    "music_audio": [
        "https://www.spotify.com",
        "https://music.apple.com",
        "https://soundcloud.com",
        "https://www.pandora.com",
        "https://music.youtube.com",
        "https://bandcamp.com",
    ],
    # Education & learning
    "education": [
        "https://www.coursera.org",
        "https://www.udemy.com",
        "https://www.khanacademy.org",
        "https://www.edx.org",
        "https://www.udacity.com",
        "https://www.duolingo.com",
        "https://www.codecademy.com",
        "https://www.skillshare.com",
        "https://brilliant.org",
    ],
    # Business & productivity
    "productivity": [
        "https://www.notion.so",
        "https://www.trello.com",
        "https://www.asana.com",
        "https://slack.com",
        "https://www.dropbox.com",
        "https://drive.google.com",
        "https://www.figma.com",
        "https://www.canva.com",
        "https://miro.com",
        "https://www.airtable.com",
    ],
    # Finance & crypto
    "finance": [
        "https://www.paypal.com",
        "https://stripe.com",
        "https://www.coinbase.com",
        "https://www.binance.com",
        "https://www.robinhood.com",
        "https://www.mint.com",
        "https://www.chase.com",
        "https://www.bankofamerica.com",
    ],
    # Travel & maps
    "travel": [
        "https://www.google.com/maps",
        "https://www.airbnb.com",
        "https://www.booking.com",
        "https://www.expedia.com",
        "https://www.tripadvisor.com",
        "https://www.kayak.com",
        "https://www.uber.com",
        "https://www.lyft.com",
    ],
    # Food delivery & recipes
    "food": [
        "https://www.doordash.com",
        "https://www.ubereats.com",
        "https://www.grubhub.com",
        "https://www.allrecipes.com",
        "https://www.foodnetwork.com",
        "https://www.epicurious.com",
        "https://www.seriouseats.com",
    ],
    # Healthcare & fitness
    "health_fitness": [
        "https://www.webmd.com",
        "https://www.mayoclinic.org",
        "https://www.healthline.com",
        "https://www.myfitnesspal.com",
        "https://www.strava.com",
        "https://www.fitbit.com",
    ],
    # Gaming
    "gaming": [
        "https://store.steampowered.com",
        "https://www.epicgames.com",
        "https://www.roblox.com",
        "https://www.minecraft.net",
        "https://www.ea.com",
        "https://www.ign.com",
        "https://www.gamespot.com",
    ],
    # Government & non-profit
    "government_nonprofit": [
        "https://www.usa.gov",
        "https://www.cdc.gov",
        "https://www.nih.gov",
        "https://www.nasa.gov",
        "https://www.un.org",
        "https://www.redcross.org",
        "https://www.wikipedia.org",
    ],
    # Design & creative
    "design_creative": [
        "https://dribbble.com",
        "https://www.behance.net",
        "https://www.deviantart.com",
        "https://unsplash.com",
        "https://www.shutterstock.com",
        "https://www.adobe.com",
    ],
}


# Sub-pages and paths to increase diversity within domains
URL_PATTERNS = {
    # Generic patterns
    "generic": [
        "",  # Homepage
        "/about",
        "/contact",
        "/pricing",
        "/blog",
        "/features",
        "/products",
        "/services",
        "/login",
        "/signup",
        "/search",
        "/help",
        "/support",
        "/faq",
        "/terms",
        "/privacy",
    ],
    # E-commerce specific
    "ecommerce": [
        "/products",
        "/categories",
        "/deals",
        "/new-arrivals",
        "/bestsellers",
        "/sale",
        "/cart",
        "/checkout",
        "/wishlist",
        "/account",
    ],
    # Social media specific
    "social": [
        "/explore",
        "/trending",
        "/notifications",
        "/messages",
        "/profile",
        "/settings",
        "/feed",
        "/discover",
    ],
    # News/blog specific
    "content": [
        "/articles",
        "/news",
        "/blog",
        "/latest",
        "/popular",
        "/category/technology",
        "/category/business",
        "/archive",
    ],
    # Documentation specific
    "docs": [
        "/docs",
        "/documentation",
        "/getting-started",
        "/api",
        "/reference",
        "/guides",
        "/tutorials",
        "/examples",
    ],
}


# Additional diverse domains by category
EXTENDED_DOMAINS = {
    "universities": [
        "https://www.harvard.edu",
        "https://www.stanford.edu",
        "https://www.mit.edu",
        "https://www.oxford.ac.uk",
        "https://www.cambridge.org",
        "https://www.berkeley.edu",
    ],
    "portfolios": [
        "https://www.behance.net",
        "https://dribbble.com",
        "https://www.artstation.com",
    ],
    "weather": [
        "https://weather.com",
        "https://www.accuweather.com",
        "https://www.wunderground.com",
    ],
    "sports": [
        "https://www.espn.com",
        "https://www.nfl.com",
        "https://www.nba.com",
        "https://www.mlb.com",
        "https://www.fifa.com",
        "https://www.olympic.org",
    ],
    "real_estate": [
        "https://www.zillow.com",
        "https://www.realtor.com",
        "https://www.redfin.com",
    ],
    "job_boards": [
        "https://www.indeed.com",
        "https://www.glassdoor.com",
        "https://www.monster.com",
        "https://www.dice.com",
        "https://www.ziprecruiter.com",
    ],
    "analytics": [
        "https://analytics.google.com",
        "https://www.hotjar.com",
        "https://mixpanel.com",
    ],
    "cloud_services": [
        "https://aws.amazon.com",
        "https://cloud.google.com",
        "https://azure.microsoft.com",
        "https://www.heroku.com",
        "https://vercel.com",
        "https://www.netlify.com",
    ],
    "communication": [
        "https://zoom.us",
        "https://meet.google.com",
        "https://www.skype.com",
        "https://www.whatsapp.com",
        "https://telegram.org",
    ],
}


def generate_urls(
    num_urls: int = 1000,
    include_subpages: bool = True,
    category_filter: List[str] = None,
    random_seed: int = 42,
) -> List[str]:
    """
    Generate diverse list of URLs.

    Args:
        num_urls: Target number of URLs to generate
        include_subpages: Whether to include sub-pages (e.g., /about, /pricing)
        category_filter: List of categories to include (None = all)
        random_seed: Random seed for reproducibility

    Returns:
        List of unique URLs
    """
    random.seed(random_seed)
    urls: Set[str] = set()

    # Collect all base URLs
    all_categories = {**SEED_URLS, **EXTENDED_DOMAINS}

    if category_filter:
        all_categories = {
            k: v for k, v in all_categories.items() if k in category_filter
        }

    base_urls = []
    for category, url_list in all_categories.items():
        for url in url_list:
            base_urls.append((url, category))

    # Shuffle for randomness
    random.shuffle(base_urls)

    # Add all base URLs first
    for url, _ in base_urls:
        urls.add(url)

    # Add sub-pages if requested
    if include_subpages and len(urls) < num_urls:
        for base_url, category in base_urls:
            if len(urls) >= num_urls:
                break

            # Determine which patterns to use based on category
            if category in ["ecommerce"]:
                patterns = URL_PATTERNS["ecommerce"]
            elif category in ["social_media"]:
                patterns = URL_PATTERNS["social"]
            elif category in ["news_media", "education"]:
                patterns = URL_PATTERNS["content"]
            elif category in ["documentation", "tech_dev"]:
                patterns = URL_PATTERNS["docs"]
            else:
                patterns = URL_PATTERNS["generic"]

            # Add sub-pages
            for path in patterns:
                if len(urls) >= num_urls:
                    break
                if path:  # Skip empty path (already added as base)
                    full_url = base_url.rstrip("/") + path
                    urls.add(full_url)

    # Convert to list and shuffle
    url_list = list(urls)
    random.shuffle(url_list)

    # Return requested number
    return url_list[:num_urls]


def generate_github_repos(num_repos: int = 100) -> List[str]:
    """Generate URLs for popular GitHub repositories."""
    # Top GitHub repos across different languages/topics
    repos = [
        # JavaScript/TypeScript
        "freeCodeCamp/freeCodeCamp",
        "microsoft/vscode",
        "facebook/react",
        "vuejs/vue",
        "angular/angular",
        "nodejs/node",
        "webpack/webpack",
        "jquery/jquery",
        "d3/d3",
        "axios/axios",
        "lodash/lodash",
        # Python
        "tensorflow/tensorflow",
        "keras-team/keras",
        "scikit-learn/scikit-learn",
        "django/django",
        "flask/flask",
        "fastapi/fastapi",
        "pandas-dev/pandas",
        "numpy/numpy",
        "matplotlib/matplotlib",
        "scrapy/scrapy",
        # Machine Learning
        "openai/gpt-3",
        "huggingface/transformers",
        "pytorch/pytorch",
        "apache/spark",
        "dmlc/xgboost",
        "mlflow/mlflow",
        # DevOps
        "kubernetes/kubernetes",
        "docker/docker-ce",
        "ansible/ansible",
        "terraform-providers/terraform-provider-aws",
        "hashicorp/terraform",
        # Go
        "golang/go",
        "gin-gonic/gin",
        "gohugoio/hugo",
        "prometheus/prometheus",
        # Rust
        "rust-lang/rust",
        "denoland/deno",
        "tokio-rs/tokio",
        # Java
        "spring-projects/spring-boot",
        "elastic/elasticsearch",
        "apache/kafka",
        # Mobile
        "flutter/flutter",
        "facebook/react-native",
        "ionic-team/ionic-framework",
        # Tools
        "git/git",
        "curl/curl",
        "vim/vim",
        "neovim/neovim",
        # Awesome lists
        "sindresorhus/awesome",
        "awesome-selfhosted/awesome-selfhosted",
        # Games
        "godotengine/godot",
        "unity-technologies/unity",
    ]

    urls = [f"https://github.com/{repo}" for repo in repos]
    return urls[:num_repos]


def generate_stackoverflow_pages(num_pages: int = 50) -> List[str]:
    """Generate Stack Overflow URLs for popular tags."""
    tags = [
        "javascript",
        "python",
        "java",
        "c#",
        "php",
        "android",
        "html",
        "css",
        "reactjs",
        "node.js",
        "c++",
        "typescript",
        "angular",
        "django",
        "sql",
        "ios",
        "swift",
        "kotlin",
        "rust",
        "go",
        "vue.js",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "machine-learning",
        "tensorflow",
        "pytorch",
        "pandas",
    ]

    urls = []
    base = "https://stackoverflow.com"

    # Main tag pages
    for tag in tags[:num_pages]:
        urls.append(f"{base}/questions/tagged/{tag}")

    return urls


def generate_wikipedia_pages(num_pages: int = 100) -> List[str]:
    """Generate Wikipedia URLs for diverse topics."""
    topics = [
        # Science
        "Artificial_intelligence",
        "Machine_learning",
        "Quantum_computing",
        "Relativity",
        "Evolution",
        "DNA",
        "Climate_change",
        "Photosynthesis",
        # Technology
        "Internet",
        "World_Wide_Web",
        "Computer",
        "Smartphone",
        "Blockchain",
        "Cryptocurrency",
        "Cloud_computing",
        "5G",
        "HTTP",
        "TCP/IP",
        # History
        "World_War_II",
        "Ancient_Rome",
        "Renaissance",
        "Industrial_Revolution",
        "Cold_War",
        "Space_Race",
        "Internet_Age",
        # Geography
        "Earth",
        "Pacific_Ocean",
        "Amazon_rainforest",
        "Sahara",
        "Mount_Everest",
        # Culture
        "Music",
        "Film",
        "Literature",
        "Art",
        "Philosophy",
        "Religion",
        # Programming
        "Python_(programming_language)",
        "JavaScript",
        "C++",
        "Java_(programming_language)",
        "Ruby_(programming_language)",
        "Go_(programming_language)",
        # Companies
        "Google",
        "Apple_Inc.",
        "Microsoft",
        "Amazon_(company)",
        "Meta_Platforms",
        "Tesla,_Inc.",
        "SpaceX",
        "Netflix",
        "Adobe_Inc.",
        # People
        "Albert_Einstein",
        "Isaac_Newton",
        "Marie_Curie",
        "Nikola_Tesla",
        "Ada_Lovelace",
        "Alan_Turing",
        "Steve_Jobs",
        "Elon_Musk",
    ]

    urls = [f"https://en.wikipedia.org/wiki/{topic}" for topic in topics]

    # Add some category pages
    categories = [
        "Computer_science",
        "Software_engineering",
        "Web_development",
        "Mobile_app_development",
        "Data_science",
        "Cryptography",
    ]
    urls.extend([f"https://en.wikipedia.org/wiki/Category:{cat}" for cat in categories])

    return urls[:num_pages]


def save_urls_to_csv(urls: List[str], output_path: str = "urls.csv"):
    """Save URLs to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for url in urls:
            writer.writerow([url])

    print(f"✅ Saved {len(urls)} URLs to {output_path}")


def generate_comprehensive_dataset(
    output_path: str = "urls.csv",
    num_general: int = 800,
    num_github: int = 100,
    num_stackoverflow: int = 50,
    num_wikipedia: int = 50,
    include_subpages: bool = True,
    random_seed: int = 42,
):
    """
    Generate comprehensive, diverse URL dataset.

    Default: 1000 total URLs with good category distribution
    """
    print("🔨 Generating comprehensive URL dataset...")
    print(
        f"   Target: {num_general + num_github + num_stackoverflow + num_wikipedia} URLs"
    )

    all_urls: Set[str] = set()

    # 1. General websites with sub-pages
    print(f"\n📋 Generating {num_general} general website URLs...")
    general_urls = generate_urls(
        num_urls=num_general, include_subpages=include_subpages, random_seed=random_seed
    )
    all_urls.update(general_urls)
    print(f"   Added {len(general_urls)} general URLs")

    # 2. GitHub repositories
    print(f"\n🐙 Generating {num_github} GitHub repository URLs...")
    github_urls = generate_github_repos(num_repos=num_github)
    all_urls.update(github_urls)
    print(f"   Added {len(github_urls)} GitHub URLs")

    # 3. Stack Overflow pages
    print(f"\n💬 Generating {num_stackoverflow} Stack Overflow URLs...")
    so_urls = generate_stackoverflow_pages(num_pages=num_stackoverflow)
    all_urls.update(so_urls)
    print(f"   Added {len(so_urls)} Stack Overflow URLs")

    # 4. Wikipedia pages
    print(f"\n📚 Generating {num_wikipedia} Wikipedia URLs...")
    wiki_urls = generate_wikipedia_pages(num_pages=num_wikipedia)
    all_urls.update(wiki_urls)
    print(f"   Added {len(wiki_urls)} Wikipedia URLs")

    # Convert to sorted list
    final_urls = sorted(list(all_urls))

    # Save to CSV
    save_urls_to_csv(final_urls, output_path)

    # Print statistics
    print("\n" + "=" * 70)
    print("📊 DATASET STATISTICS")
    print("=" * 70)
    print(f"Total unique URLs: {len(final_urls)}")

    # Count by domain
    from collections import Counter
    from urllib.parse import urlparse

    domains = Counter()
    for url in final_urls:
        domain = urlparse(url).netloc
        domains[domain] += 1

    print(f"\nTop 10 domains:")
    for domain, count in domains.most_common(10):
        print(f"  {domain:30s}: {count:3d} URLs")

    print(f"\nTotal domains: {len(domains)}")
    print(f"Output file: {output_path}")
    print("=" * 70)

    return final_urls


def main():
    """Main CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate diverse URL dataset for web screenshot training"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="urls.csv",
        help="Output CSV file path (default: urls.csv)",
    )
    parser.add_argument(
        "--num-urls",
        "-n",
        type=int,
        default=1000,
        help="Total number of URLs to generate (default: 1000)",
    )
    parser.add_argument(
        "--general",
        type=int,
        default=None,
        help="Number of general website URLs (default: 80%% of total)",
    )
    parser.add_argument(
        "--github",
        type=int,
        default=None,
        help="Number of GitHub URLs (default: 10%% of total)",
    )
    parser.add_argument(
        "--stackoverflow",
        type=int,
        default=None,
        help="Number of Stack Overflow URLs (default: 5%% of total)",
    )
    parser.add_argument(
        "--wikipedia",
        type=int,
        default=None,
        help="Number of Wikipedia URLs (default: 5%% of total)",
    )
    parser.add_argument(
        "--no-subpages",
        action="store_true",
        help="Don't include sub-pages (only homepages)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Calculate defaults if not specified
    num_general = args.general if args.general else int(args.num_urls * 0.8)
    num_github = args.github if args.github else int(args.num_urls * 0.1)
    num_stackoverflow = (
        args.stackoverflow if args.stackoverflow else int(args.num_urls * 0.05)
    )
    num_wikipedia = args.wikipedia if args.wikipedia else int(args.num_urls * 0.05)

    generate_comprehensive_dataset(
        output_path=args.output,
        num_general=num_general,
        num_github=num_github,
        num_stackoverflow=num_stackoverflow,
        num_wikipedia=num_wikipedia,
        include_subpages=not args.no_subpages,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
