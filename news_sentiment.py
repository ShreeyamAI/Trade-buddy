"""
News Sentiment Module for Trade Buddy.
Fetches Indian financial markets news from public RSS feeds.
Uses keyword-based sentiment scoring to assess market mood.

No API key required. No web scraping.
RSS feeds are publicly published and designed for programmatic access.
Sentiment is an experimental signal only — not financial advice.
"""

import requests
import xml.etree.ElementTree as ET
import re
from datetime import datetime
from typing import List, Dict, Any

# ── Public Indian Markets RSS Feeds ──────────────────────
NEWS_SOURCES = [
    {"name": "Economic Times Markets", "url": "https://economictimes.indiatimes.com/markets/rss.cms", "timeout": 7},
    {"name": "Business Standard Markets", "url": "https://www.business-standard.com/rss/markets-106.rss", "timeout": 7},
    {"name": "Moneycontrol Markets", "url": "https://www.moneycontrol.com/rss/results.xml", "timeout": 7},
    {"name": "LiveMint Markets", "url": "https://www.livemint.com/rss/money", "timeout": 7},
]

# Sentiment keyword sets
BULLISH_WORDS = {
    'rise', 'rises', 'rally', 'rallies', 'rallied', 'gain', 'gains',
    'surge', 'surges', 'surged', 'advance', 'advances', 'climb',
    'climbs', 'jumped', 'soar', 'soars', 'recovery', 'recovers',
    'rebound', 'bounced', 'breakout', 'strong', 'positive', 'upward',
    'upgrade', 'upbeat', 'optimistic', 'inflow', 'profit', 'record high',
    'buying', 'accumulate', 'outperform', 'expansion', 'boost',
    'improvement', 'robust', 'green', 'growth', 'high', 'support',
}

BEARISH_WORDS = {
    'fall', 'falls', 'fell', 'drop', 'drops', 'dropped', 'decline',
    'declines', 'declined', 'bear', 'bearish', 'loss', 'losses',
    'plunge', 'plunges', 'plunged', 'slump', 'slumps', 'slumped',
    'crash', 'crashing', 'correction', 'breakdown', 'weak', 'weakness',
    'negative', 'downward', 'selling', 'sell-off', 'outflow', 'fear',
    'concern', 'caution', 'uncertainty', 'volatility', 'risk', 'warning',
    'recession', 'slowdown', 'pressure', 'drag', 'underperform', 'red',
    'inflation', 'rate hike', 'geopolitical', 'crisis', 'default',
}

# NIFTY / India market relevance filter
NIFTY_WORDS = {
    'nifty', 'sensex', 'nse', 'bse', 'indian market', 'india market',
    'dalal street', 'equity', 'stock market', 'index', 'indices', 'benchmark',
    'nifty 50', 'nifty50', 'market open', 'market close', 'midcap', 'smallcap',
}


def _score_text(text: str) -> float:
    """
    Returns a sentiment score from -1.0 (very bearish) to +1.0 (very bullish)
    based on keyword counting.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    text_lower = text.lower()

    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


def _is_relevant(text: str) -> bool:
    """Returns True if the article text mentions NIFTY/Indian markets."""
    t = text.lower()
    return any(kw in t for kw in NIFTY_WORDS)


def _clean_html(raw: str) -> str:
    """Strips HTML tags and trims whitespace from text."""
    return re.sub(r'<[^>]+>', '', raw or '').strip()


def _parse_rss(xml_text: str, source_name: str, max_items: int = 8) -> List[Dict[str, Any]]:
    """Parses a raw RSS XML string into a list of article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        # RSS 2.0 uses <item>, Atom uses <entry>
        items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
        for item in items[:max_items * 2]:
            def text(tag):
                el = item.find(tag) or item.find(f'{{http://www.w3.org/2005/Atom}}{tag}')
                return (el.text or '') if el is not None else ''

            title = _clean_html(text('title')) or 'Untitled'
            desc = _clean_html(text('description') or text('summary'))[:400]
            pub = text('pubDate') or text('published') or ''

            link_el = item.find('link') or item.find('{http://www.w3.org/2005/Atom}link')
            link = ''
            if link_el is not None:
                link = link_el.text or link_el.get('href', '')

            full = f"{title} {desc}"
            score = _score_text(full)
            relevant = _is_relevant(full)

            articles.append({
                "source": source_name,
                "title": title,
                "summary": desc,
                "pub_date": pub,
                "link": link.strip(),
                "sentiment": score,
                "sentiment_label": (
                    "🟢 Bullish" if score > 0.1 else
                    "🔴 Bearish" if score < -0.1 else
                    "⚪ Neutral"
                ),
                "relevant": relevant,
            })
            if len(articles) >= max_items:
                break
    except Exception:
        pass
    return articles


def fetch_news(max_per_source: int = 6) -> Dict[str, Any]:
    """
    Fetches latest headlines from all RSS sources.
    Returns articles sorted by market relevance, aggregate sentiment score,
    and per-source fetch status.
    """
    all_articles: List[Dict] = []
    errors: List[str] = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; TradeBuddy/1.0)',
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
    }

    for src in NEWS_SOURCES:
        try:
            r = requests.get(src['url'], timeout=src['timeout'], headers=headers)
            if r.status_code == 200:
                arts = _parse_rss(r.text, src['name'], max_per_source)
                all_articles.extend(arts)
            else:
                errors.append(f"{src['name']}: HTTP {r.status_code}")
        except requests.exceptions.Timeout:
            errors.append(f"{src['name']}: Timed out")
        except requests.exceptions.ConnectionError:
            errors.append(f"{src['name']}: Connection failed")
        except Exception as e:
            errors.append(f"{src['name']}: {str(e)[:60]}")

    if not all_articles:
        return {
            "articles": [],
            "aggregate_score": 0.0,
            "market_sentiment": "neutral",
            "sentiment_label": "⚪ Neutral",
            "total_articles": 0,
            "relevant_count": 0,
            "errors": errors,
            "fetched_at": datetime.now().strftime('%H:%M:%S'),
        }

    # Sort: market-relevant articles first
    all_articles.sort(key=lambda a: (not a['relevant'], a.get('pub_date', '')))

    relevant = [a for a in all_articles if a['relevant']]
    scores = [a['sentiment'] for a in (relevant or all_articles)]
    agg = round(float(sum(scores) / len(scores)), 3) if scores else 0.0

    if agg > 0.12:
        sentiment, label = "bullish", "🟢 Bullish"
    elif agg < -0.12:
        sentiment, label = "bearish", "🔴 Bearish"
    else:
        sentiment, label = "neutral", "⚪ Neutral"

    return {
        "articles": all_articles,
        "aggregate_score": agg,
        "market_sentiment": sentiment,
        "sentiment_label": label,
        "total_articles": len(all_articles),
        "relevant_count": len(relevant),
        "errors": errors,
        "fetched_at": datetime.now().strftime('%H:%M:%S'),
    }
