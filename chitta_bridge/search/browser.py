"""Cloudflare-aware fetch: lightweight by default, browser only when forced to.

Architecture — browser as a rare cookie-minter, not a per-request engine:
  1. Try curl_cffi (Firefox JA3 impersonation, a few MB). Most sites pass, no browser.
  2. If Cloudflare challenges it, mint a cf_clearance cookie ONCE with camoufox
     (a fortified Firefox — the only thing that clears CF's managed JS challenge
     headlessly), cache {cookies, UA} per domain (~25 min TTL).
  3. Retry curl_cffi with the cookie; reuse it for all later fetches to that domain.

So camoufox (heavy, ~700 MB, requires playwright==1.49) fires at most once per
domain per TTL; everything else rides the tiny client. `render=True` forces a full
camoufox load for genuinely JS-rendered (SPA) pages.
"""

from __future__ import annotations

import html as _html
import importlib.util
import re
import time
from urllib.parse import urlsplit

__all__ = ["BrowserFetch", "BrowserStackUnavailable"]


class BrowserStackUnavailable(RuntimeError):
    """The optional `browser` extra (curl_cffi) isn't installed.

    Caller should fall back to the normal web_fetch route.
    """


def _installed(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

_CHALLENGE_MARKERS = (
    "just a moment", "checking your browser", "performing security verification",
    "verifying you are human", "cf-challenge", "challenge-platform",
)
_SUPP_PATS = (
    "downloadsupplement", "/suppl", "supinfo", "media/", ".xlsx", ".xls", ".csv",
    ".tsv", ".zip", ".pdf", ".docx", "figshare", "zenodo", "datadryad",
)
_IMPERSONATE = "firefox135"
_TTL_S = 1500  # cf_clearance is good ~30 min; refresh a little early.
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)


class BrowserFetch:
    # domain -> {"cookies": dict, "ua": str, "ts": float}
    _cache: dict[str, dict] = {}

    # ── capability probes ────────────────────────────────────────────────
    @staticmethod
    def available() -> bool:
        """Minimum stack for any lightweight fetch (Firefox-JA3 client)."""
        return _installed("curl_cffi")

    @staticmethod
    def can_mint() -> bool:
        """camoufox present → can clear Cloudflare's managed JS challenge."""
        return _installed("camoufox")

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _root(url: str) -> str:
        s = urlsplit(url)
        return f"{s.scheme}://{s.netloc}/"

    @classmethod
    def _fresh(cls, domain: str):
        e = cls._cache.get(domain)
        if e and time.monotonic() - e["ts"] < _TTL_S:
            return e
        return None

    @staticmethod
    def _looks_challenged(status: int, headers, text: str) -> bool:
        if status in (403, 429, 503):
            return True
        if "cf-mitigated" in {k.lower() for k in headers}:
            return True
        if "html" in headers.get("content-type", "").lower():
            low = text[:5000].lower()
            return any(m in low for m in _CHALLENGE_MARKERS)
        return False

    @staticmethod
    def _strip_html(body: str, max_chars: int) -> str:
        body = _TAG_RE.sub(" ", body)
        body = re.sub(r"<[^>]+>", " ", body)
        body = _html.unescape(re.sub(r"\s+", " ", body)).strip()
        return body[:max_chars]

    @classmethod
    def _supp_links(cls, body: str, base: str) -> list[str]:
        seen, out = set(), []
        for h in _HREF_RE.findall(body):
            full = h if h.startswith("http") else base.rstrip("/") + "/" + h.lstrip("/")
            if any(p in full.lower() for p in _SUPP_PATS) and full not in seen:
                seen.add(full)
                out.append(full)
        return out[:40]

    # ── heavy step: mint a cf_clearance cookie via camoufox ──────────────
    @classmethod
    def _mint(cls, url: str) -> dict:
        from camoufox.sync_api import Camoufox

        domain = urlsplit(url).netloc
        with Camoufox(headless=True, geoip=True) as b:
            page = b.new_page(no_viewport=True)
            page.goto(cls._root(url), wait_until="domcontentloaded", timeout=90000)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                blob = ((page.title() or "") + " " + page.inner_text("body")[:300]).lower()
                if not any(m in blob for m in _CHALLENGE_MARKERS):
                    break
                page.wait_for_timeout(1500)
            page.wait_for_timeout(1500)
            ua = page.evaluate("() => navigator.userAgent")
            cookies = {
                c["name"]: c["value"]
                for c in page.context.cookies()
                if c["domain"].lstrip(".").endswith(".".join(domain.split(".")[-2:]))
            }
        entry = {"cookies": cookies, "ua": ua, "ts": time.monotonic()}
        cls._cache[domain] = entry
        return entry

    # ── full browser render (SPA / JS-only pages) ────────────────────────
    @classmethod
    def _render(cls, url: str, max_chars: int, wait_selector: str | None) -> str:
        from camoufox.sync_api import Camoufox

        with Camoufox(headless=True, geoip=True) as b:
            page = b.new_page(no_viewport=True)
            page.goto(url, wait_until="domcontentloaded", timeout=90000)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                blob = ((page.title() or "") + " " + page.inner_text("body")[:300]).lower()
                if not any(m in blob for m in _CHALLENGE_MARKERS):
                    break
                page.wait_for_timeout(1500)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=90000)
            text = page.inner_text("body")[:max_chars]
            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        links = [h for h in hrefs if any(p in h.lower() for p in _SUPP_PATS)][:40]
        if links:
            text += "\n\n--- candidate download links ---\n" + "\n".join(links)
        return text

    # ── public entry point ───────────────────────────────────────────────
    @classmethod
    def fetch(
        cls,
        url: str,
        max_chars: int = 20000,
        download_path: str | None = None,
        wait_selector: str | None = None,
        render: bool = False,
    ) -> str:
        if not cls.available():
            raise BrowserStackUnavailable

        if render:
            if not cls.can_mint():
                raise BrowserStackUnavailable
            return cls._render(url, max_chars, wait_selector)

        from curl_cffi import requests as cc

        domain = urlsplit(url).netloc

        def attempt(entry):
            headers = {"Referer": cls._root(url), "Accept": "*/*",
                       "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "no-cors",
                       "Sec-Fetch-Dest": "empty"}
            cookies = {}
            if entry:
                headers["User-Agent"] = entry["ua"]
                cookies = entry["cookies"]
            return cc.get(url, headers=headers, cookies=cookies,
                          impersonate=_IMPERSONATE, timeout=90, allow_redirects=True)

        entry = cls._fresh(domain)
        try:
            r = attempt(entry)
        except Exception:
            r = None

        if r is None or cls._looks_challenged(
                r.status_code, r.headers, "" if download_path else r.text):
            if cls.can_mint():
                entry = cls._mint(url)
                r = attempt(entry)
            elif r is None:
                raise BrowserStackUnavailable

        ct = r.headers.get("content-type", "").lower()
        if download_path:
            if r.status_code == 200 and "html" not in ct:
                with open(download_path, "wb") as f:
                    f.write(r.content)
                return f"saved -> {download_path} ({len(r.content)} bytes, {ct})"
            return f"(download failed: HTTP {r.status_code} {ct} for {url})"

        text = cls._strip_html(r.text, max_chars)
        links = cls._supp_links(r.text, cls._root(url))
        if links:
            text += "\n\n--- candidate download links ---\n" + "\n".join(links)
        return text
