"""Event-driven crawler that respects robots.txt, dedups, and emits clean shards."""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.robotparser
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests
import trafilatura
from bs4 import BeautifulSoup
from langdetect import LangDetectException, detect
from simhash import Simhash

from ..utils.config import load_config


def _normalize_domain(domain: str) -> str:
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _canonical_url(url: str) -> Optional[str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        if not parsed.scheme and parsed.netloc:
            parsed = parsed._replace(scheme="https")
        else:
            return None
    if not parsed.netloc:
        return None
    path = parsed.path or "/"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


@dataclass
class CrawlSettings:
    output_dir: Path
    shard_size: int = 200
    max_pages: int = 1000
    per_domain_quota: int = 250
    rate_limit_seconds: float = 1.0
    user_agent: str = "HatchlingCrawler/0.1 (+https://github.com/inclusionai/baby-hatchling)"
    english_only: bool = True
    min_tokens: int = 120
    max_tokens: int = 4096
    allow_domains: List[str] = field(default_factory=list)
    deny_patterns: List[str] = field(default_factory=list)
    license_keywords: List[str] = field(default_factory=list)
    seeds: List[str] = field(default_factory=list)
    simhash_threshold: int = 3


class HatchlingCrawler:
    def __init__(self, cfg: CrawlSettings) -> None:
        self.cfg = cfg
        self.headers = {"User-Agent": cfg.user_agent}
        self.queue: deque[str] = deque()
        self.visited: Set[str] = set()
        self.domain_counts: Counter[str] = Counter()
        self.robots: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self.simhashes: List[Simhash] = []
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self.current_shard_count = 0
        self.stats = Counter()
        for seed in cfg.seeds:
            canonical = _canonical_url(seed)
            if canonical:
                self.queue.append(canonical)

    def _domain_allowed(self, domain: str) -> bool:
        if not self.cfg.allow_domains:
            return True
        domain = _normalize_domain(domain)
        for allowed in self.cfg.allow_domains:
            allowed = _normalize_domain(allowed)
            if domain == allowed or domain.endswith("." + allowed):
                return True
        return False

    def _denied(self, url: str) -> bool:
        for pattern in self.cfg.deny_patterns:
            if pattern and pattern in url:
                return True
        return False

    def _can_fetch(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        if not self._domain_allowed(domain):
            return False
        if self._denied(url):
            return False
        parser = self.robots.get(domain)
        if parser is None:
            parser = urllib.robotparser.RobotFileParser()
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            try:
                parser.set_url(robots_url)
                parser.read()
            except Exception:
                parser = urllib.robotparser.RobotFileParser()
            self.robots[domain] = parser
        return parser.can_fetch(self.cfg.user_agent, url)

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
        except requests.RequestException:
            self.stats["fetch_error"] += 1
            return None
        if resp.status_code != 200:
            self.stats["fetch_error"] += 1
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            self.stats["skipped_content_type"] += 1
            return None
        return resp.text

    def _extract_text(self, html: str) -> str:
        text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
        return text or ""

    def _is_english(self, text: str) -> bool:
        if not self.cfg.english_only:
            return True
        chunk = text[:1000]
        if not chunk.strip():
            return False
        try:
            return detect(chunk) == "en"
        except LangDetectException:
            return False

    def _passes_license(self, text: str, html: str) -> bool:
        if not self.cfg.license_keywords:
            return True
        haystack = f"{text}\n{html}".lower()
        for keyword in self.cfg.license_keywords:
            if keyword.lower() in haystack:
                return True
        return False

    def _is_duplicate(self, text: str) -> bool:
        signature = Simhash(text)
        for existing in self.simhashes[-5000:]:
            if existing.distance(signature) <= self.cfg.simhash_threshold:
                return True
        self.simhashes.append(signature)
        return False

    def _write_record(self, record: dict) -> None:
        shard_path = self.output_dir / f"shard_{self.shard_idx:04d}.jsonl"
        with shard_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.current_shard_count += 1
        if self.current_shard_count >= self.cfg.shard_size:
            self.shard_idx += 1
            self.current_shard_count = 0

    def _enqueue_links(self, html: str, base_url: str) -> None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            joined = urllib.parse.urljoin(base_url, href)
            canonical = _canonical_url(joined)
            if not canonical or canonical in self.visited:
                continue
            if self._denied(canonical):
                continue
            self.queue.append(canonical)

    def crawl(self) -> Counter:
        start_time = time.time()
        while self.queue and self.stats["downloaded"] < self.cfg.max_pages:
            url = self.queue.popleft()
            if url in self.visited:
                continue
            self.visited.add(url)
            if not self._can_fetch(url):
                self.stats["skipped_policy"] += 1
                continue
            domain = urllib.parse.urlparse(url).netloc.lower()
            if self.domain_counts[domain] >= self.cfg.per_domain_quota:
                self.stats["skipped_quota"] += 1
                continue
            time.sleep(self.cfg.rate_limit_seconds)
            html = self._fetch_html(url)
            if not html:
                continue
            self.domain_counts[domain] += 1
            self._enqueue_links(html, url)
            text = self._extract_text(html)
            tokens = text.split()
            token_count = len(tokens)
            if token_count < self.cfg.min_tokens:
                self.stats["skipped_short"] += 1
                continue
            if token_count > self.cfg.max_tokens:
                text = " ".join(tokens[: self.cfg.max_tokens])
                token_count = self.cfg.max_tokens
            if not self._is_english(text):
                self.stats["skipped_language"] += 1
                continue
            if not self._passes_license(text, html):
                self.stats["skipped_license"] += 1
                continue
            if self._is_duplicate(text):
                self.stats["skipped_dup"] += 1
                continue
            record = {
                "url": url,
                "domain": domain,
                "timestamp": time.time(),
                "text": text,
                "token_count": token_count,
            }
            self._write_record(record)
            self.stats["downloaded"] += 1
        duration = max(1.0, time.time() - start_time)
        self.stats["elapsed_sec"] = int(duration)
        return self.stats


def _prepare_settings(cfg: dict) -> CrawlSettings:
    seeds = cfg.get("seeds", [])
    normalized_seeds: List[str] = []
    allow_domains = cfg.get("allow_domains", []) or []
    if isinstance(seeds, list):
        for item in seeds:
            if isinstance(item, str):
                normalized_seeds.append(item)
            elif isinstance(item, dict):
                url = item.get("url")
                if url:
                    normalized_seeds.append(url)
                allow = item.get("allow")
                if allow:
                    allow_domains.extend(allow if isinstance(allow, list) else [allow])
    settings = CrawlSettings(
        output_dir=Path(cfg.get("output_dir", "data/crawler/shards")),
        shard_size=int(cfg.get("shard_size", 200)),
        max_pages=int(cfg.get("max_pages", 1000)),
        per_domain_quota=int(cfg.get("per_domain_quota", 250)),
        rate_limit_seconds=float(cfg.get("rate_limit_seconds", 1.0)),
        user_agent=cfg.get("user_agent", CrawlSettings.user_agent),
        english_only=bool(cfg.get("english_only", True)),
        min_tokens=int(cfg.get("min_tokens", 120)),
        max_tokens=int(cfg.get("max_tokens", 4096)),
        allow_domains=allow_domains,
        deny_patterns=cfg.get("deny_patterns", []),
        license_keywords=cfg.get("license_keywords", []),
        seeds=normalized_seeds,
        simhash_threshold=int(cfg.get("simhash_threshold", 3)),
    )
    if not settings.seeds:
        raise ValueError("Crawler configuration must provide at least one seed URL.")
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Hatchling-NEURO crawler")
    parser.add_argument("--config", default="configs/crawler_english.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = _prepare_settings(cfg)
    crawler = HatchlingCrawler(settings)
    stats = crawler.crawl()
    print("Crawl summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
