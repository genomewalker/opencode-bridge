"""Web search and page-fetch utilities (DuckDuckGo + academic API router)."""

from __future__ import annotations

import html as _html
import json as _json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

__all__ = ["WebSearch"]


class WebSearch:
    """Search the web via DuckDuckGo HTML and return parsed results."""

    _DDG_URL = "https://html.duckduckgo.com/html/"
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    _RESULT_RE = re.compile(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?<a class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    @classmethod
    def search(cls, query: str, max_results: int = 8, timeout: int = 10) -> list[dict]:
        data = urllib.parse.urlencode({"q": query}).encode()
        req = urllib.request.Request(cls._DDG_URL, data=data, headers=cls._HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        results = []
        for url, title, snippet in cls._RESULT_RE.findall(body):
            if "/y.js?" in url:
                # DuckDuckGo redirect — extract actual URL
                m = re.search(r"uddg=([^&]+)", url)
                if m:
                    url = urllib.parse.unquote(m.group(1))
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            title = _html.unescape(title)
            snippet = _html.unescape(snippet)
            if url and title:
                results.append({"url": url, "title": title, "snippet": snippet})
            if len(results) >= max_results:
                break
        return results

    @classmethod
    def search_formatted(cls, query: str, max_results: int = 8) -> str:
        results = cls.search(query, max_results)
        if not results:
            return f"No results found for: {query}"
        lines = [f"Web search: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def fetch_page(cls, url: str, max_chars: int = 12000, timeout: int = 15) -> str:
        # ── Academic URL router (bypasses Cloudflare on preprint servers) ──
        academic = cls._academic_fetch(url, timeout=timeout)
        if academic:
            return academic[:max_chars]

        # ── General fetch with browser-like headers ────────────────────────
        # r.jina.ai returns plain markdown — skip encoding negotiation so we
        # get raw text instead of brotli/gzip that urllib can't decompress.
        jina = "r.jina.ai" in url
        headers = {
            **cls._HEADERS,
            "Accept": "text/plain" if jina else "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            **({"Cache-Control": "no-cache", "Pragma": "no-cache"} if not jina else {}),
            **({} if jina else {"Accept-Encoding": "gzip, deflate, br"}),
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    import gzip as _gzip
                    raw = _gzip.decompress(raw)
                enc = resp.headers.get_content_charset("utf-8")
                body = raw.decode(enc, errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                return cls._curl_fetch(url, max_chars=max_chars, timeout=timeout)
            raise
        return cls._parse_body(raw, body, url, max_chars)

    @classmethod
    def _curl_fetch(cls, url: str, max_chars: int = 12000, timeout: int = 30) -> str:
        """curl -sL fallback for Cloudflare-protected pages and direct PDF URLs."""
        import subprocess
        import hashlib

        tmp_dir = "/projects/caeg/scratch/kbd606/tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        tmp_path = os.path.join(tmp_dir, f"curl_fetch_{url_hash}")

        try:
            result = subprocess.run(
                ["curl", "-sL", "-A", "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
                 "-o", tmp_path, "-w", "%{content_type}\n%{http_code}", url],
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "(curl fallback failed — binary not found or timed out)"

        parts = result.stdout.strip().rsplit("\n", 1)
        content_type = parts[0] if len(parts) == 2 else ""
        http_code = parts[-1]

        if http_code not in ("200", ""):
            return f"(curl fallback: HTTP {http_code})"

        if not os.path.exists(tmp_path):
            return "(curl fallback: no output file)"

        # PDF: extract text via pdf_read tool path
        is_pdf = "pdf" in content_type.lower() or tmp_path.endswith(".pdf")
        with open(tmp_path, "rb") as f:
            header = f.read(5)
        if not is_pdf and header == b"%PDF-":
            is_pdf = True

        if is_pdf:
            pdf_path = tmp_path + ".pdf"
            os.rename(tmp_path, pdf_path)
            try:
                import pdfplumber
                parts: list[str] = []
                with pdfplumber.open(pdf_path) as pdf:
                    for pg in pdf.pages[:50]:
                        t = pg.extract_text(x_tolerance=2, y_tolerance=2) or ""
                        if t.strip():
                            parts.append(t.strip())
                text = "\n\n".join(parts)
            except Exception:
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(pdf_path)
                    text = "\n\n".join(
                        (p.extract_text() or "").strip()
                        for p in reader.pages[:50]
                    )
                except Exception as exc:
                    return f"(curl fetched PDF at {pdf_path} but extraction failed: {exc})"
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[truncated]"
            return text

        # HTML / plain text
        try:
            with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                body = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        text = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = _html.unescape(text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[truncated]"
        return text

    @classmethod
    def _parse_body(cls, raw: bytes, body: str, url: str, max_chars: int) -> str:
        # If server returned binary (e.g. PDF without Content-Type header), fall back to curl
        if raw[:5] == b"%PDF-":
            return cls._curl_fetch(url, max_chars=max_chars)
        text = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = _html.unescape(text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[truncated]"
        return text

    _DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")

    @classmethod
    def _clean_doi(cls, doi: str) -> str:
        """Validate a DOI for safe interpolation into API URLs ('' if unsafe)."""
        doi = doi.strip()
        if not cls._DOI_RE.match(doi) or "/.." in doi or "../" in doi:
            return ""
        return doi

    @classmethod
    def _academic_fetch(cls, url: str, timeout: int = 15) -> str:
        """Route known academic URLs to their open APIs. Returns "" if not matched."""

        # ── bioRxiv / medRxiv ─────────────────────────────────────────────
        m = re.match(
            r"https?://(?:www\.)?(biorxiv|medrxiv)\.org/content/([^?\s]+?)(?:v\d+)?(?:\.full(?:\.pdf)?|\.abstract)?/?$",
            url,
        )
        if m and cls._clean_doi(m.group(2)):
            server, doi = m.group(1), cls._clean_doi(m.group(2))
            api = f"https://api.biorxiv.org/details/{server}/{doi}/na/json"
            try:
                req = urllib.request.Request(api, headers=cls._HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read())
                items = data.get("collection", [])
                if items:
                    p = items[-1]  # latest version
                    pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
                    lines = [
                        f"# {p.get('title', 'Untitled')}",
                        f"**Authors:** {p.get('authors', '')}",
                        f"**Date:** {p.get('date', '')}  **Version:** {p.get('version', '')}",
                        f"**DOI:** {p.get('doi', '')}  **Category:** {p.get('category', '')}",
                        f"**License:** {p.get('license', '')}",
                        "",
                        "## Abstract",
                        p.get("abstract", "(no abstract)"),
                        "",
                        f"**PDF:** {pdf_url}",
                        f"**Source XML:** {p.get('jatsxml', '')}",
                    ]
                    return "\n".join(lines)
            except Exception:
                pass

        # ── arXiv ─────────────────────────────────────────────────────────
        m = re.match(r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(\S+?)(?:v\d+)?(?:\.pdf)?/?$", url)
        if m:
            arxiv_id = m.group(1)
            api = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
            try:
                req = urllib.request.Request(api, headers=cls._HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                title = re.search(r"<title>([^<]+)</title>", body)
                authors = re.findall(r"<name>([^<]+)</name>", body)
                summary = re.search(r"<summary>(.*?)</summary>", body, re.DOTALL)
                published = re.search(r"<published>([^<]+)</published>", body)
                lines = [
                    f"# {_html.unescape(title.group(1).strip()) if title else arxiv_id}",
                    f"**Authors:** {'; '.join(authors)}",
                    f"**Published:** {published.group(1)[:10] if published else ''}",
                    f"**arXiv:** https://arxiv.org/abs/{arxiv_id}",
                    "",
                    "## Abstract",
                    _html.unescape(re.sub(r"\s+", " ", summary.group(1)).strip()) if summary else "(no abstract)",
                    "",
                    f"**PDF:** https://arxiv.org/pdf/{arxiv_id}.pdf",
                ]
                return "\n".join(lines)
            except Exception:
                pass

        # ── Zenodo ───────────────────────────────────────────────────────
        m = re.match(r"https?://zenodo\.org/(?:records?|deposit)/(\d+)", url)
        if not m:
            m = re.match(r"https?://doi\.org/10\.5281/zenodo\.(\d+)", url)
        if m:
            record_id = m.group(1)
            try:
                api = f"https://zenodo.org/api/records/{record_id}"
                req = urllib.request.Request(api, headers=cls._HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read())
                meta = data.get("metadata", {})
                files = data.get("files", [])
                lines = [
                    f"# {meta.get('title', record_id)}",
                    f"**DOI:** {meta.get('doi', '')}  **Type:** {meta.get('resource_type', {}).get('type', '')}",
                    f"**Authors:** {'; '.join(a.get('name','') for a in (meta.get('creators') or [])[:6])}",
                    f"**Date:** {meta.get('publication_date', '')}  **License:** {(meta.get('license') or {}).get('id','')}",
                    "",
                    "## Description",
                    re.sub(r"<[^>]+>", "", meta.get("description", "(none)")),
                    "",
                    "## Files",
                ]
                for f in files:
                    key = f.get("key", "")
                    size = f.get("size", 0)
                    link = f.get("links", {}).get("self", "")
                    lines.append(f"- [{key}]({link}) ({size:,} bytes)")
                return "\n".join(lines)
            except Exception:
                pass

        # ── Figshare ──────────────────────────────────────────────────────
        m = re.match(r"https?://(?:figshare\.com/articles/[^/]+/[^/]+/(\d+)|doi\.org/10\.6084/m9\.figshare\.(\d+))", url)
        if m:
            article_id = m.group(1) or m.group(2)
            try:
                api = f"https://api.figshare.com/v2/articles/{article_id}"
                req = urllib.request.Request(api, headers=cls._HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read())
                lines = [
                    f"# {data.get('title', article_id)}",
                    f"**DOI:** {data.get('doi', '')}  **Type:** {data.get('defined_type_name', '')}",
                    f"**Authors:** {'; '.join(a.get('full_name','') for a in (data.get('authors') or [])[:6])}",
                    f"**Published:** {data.get('published_date', '')}",
                    "",
                    "## Description",
                    re.sub(r"<[^>]+>", "", data.get("description", "(none)")),
                    "",
                    "## Files",
                ]
                for f in data.get("files", []):
                    lines.append(f"- [{f.get('name','')}]({f.get('download_url','')}) ({f.get('size',0):,} bytes)")
                return "\n".join(lines)
            except Exception:
                pass

        # ── GitHub ────────────────────────────────────────────────────────
        m = re.match(r"https?://github\.com/([^/]+/[^/]+)(?:/tree/([^/]+)(/.*)?)?$", url)
        if m:
            repo, branch = m.group(1), m.group(2) or "HEAD"
            try:
                # Repo metadata
                api = f"https://api.github.com/repos/{repo}"
                req = urllib.request.Request(api, headers={**cls._HEADERS, "Accept": "application/vnd.github+json"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read())
                readme_url = f"https://raw.githubusercontent.com/{repo}/{branch}/README.md"
                readme = ""
                try:
                    req2 = urllib.request.Request(readme_url, headers=cls._HEADERS)
                    with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                        readme = resp2.read().decode("utf-8", errors="replace")[:3000]
                except Exception:
                    pass
                lines = [
                    f"# {data.get('full_name', repo)}",
                    f"**Description:** {data.get('description', '')}",
                    f"**Stars:** {data.get('stargazers_count', 0)}  **Language:** {data.get('language', '')}",
                    f"**License:** {(data.get('license') or {}).get('spdx_id', '')}",
                    f"**Last push:** {data.get('pushed_at', '')[:10]}",
                    f"**URL:** {data.get('html_url', '')}",
                ]
                if readme:
                    lines += ["", "## README", readme]
                return "\n".join(lines)
            except Exception:
                pass

        # ── PubMed ───────────────────────────────────────────────────────
        m = re.match(r"https?://(?:www\.)?(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/(\d+)", url)
        if m:
            pmid = m.group(1)
            try:
                api = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                       f"?db=pubmed&id={pmid}&retmode=json")
                req = urllib.request.Request(api, headers=cls._HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read())
                doc = data.get("result", {}).get(pmid, {})
                # Fetch abstract separately
                abs_api = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                           f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text")
                req2 = urllib.request.Request(abs_api, headers=cls._HEADERS)
                with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                    abstract_text = resp2.read().decode("utf-8", errors="replace")
                authors = [a.get("name", "") for a in (doc.get("authors") or [])[:6]]
                lines = [
                    f"# {doc.get('title', pmid)}",
                    f"**PMID:** {pmid}  **Journal:** {doc.get('source', '')}  **Date:** {doc.get('pubdate', '')}",
                    f"**Authors:** {'; '.join(authors)}",
                    "",
                    "## Abstract",
                    abstract_text[:4000],
                ]
                return "\n".join(lines)
            except Exception:
                pass

        # ── DOI URL → Unpaywall → OpenAlex ───────────────────────────────
        doi = None
        m = re.match(r"https?://doi\.org/(10\.\S+)", url)
        if m:
            doi = m.group(1)
        if not doi:
            m = re.search(r"(10\.\d{4,}/\S+)", url)
            if m:
                doi = m.group(1).rstrip("/")
        if doi:
            doi = cls._clean_doi(doi) or None

        if doi:
            # Try bioRxiv API for any bioRxiv-style DOI before generic handlers
            for _server in ("biorxiv", "medrxiv"):
                try:
                    api = f"https://api.biorxiv.org/details/{_server}/{doi}/na/json"
                    req = urllib.request.Request(api, headers=cls._HEADERS)
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        bdata = _json.loads(resp.read())
                    items = bdata.get("collection", [])
                    if items:
                        p = items[-1]
                        if p.get("abstract"):
                            pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
                            lines = [
                                f"# {p.get('title', 'Untitled')}",
                                f"**Authors:** {p.get('authors', '')}",
                                f"**Date:** {p.get('date', '')}  **Version:** {p.get('version', '')}",
                                f"**DOI:** {p.get('doi', '')}  **Category:** {p.get('category', '')}",
                                f"**License:** {p.get('license', '')}",
                                "", "## Abstract", p.get("abstract", ""),
                                "", f"**PDF:** {pdf_url}",
                            ]
                            return "\n".join(lines)
                except Exception:
                    pass

            # Unpaywall — good for PDF URL, but may lack abstract
            pdf_url_unpaywall = ""
            unpaywall_lines: list[str] = []
            try:
                api = f"https://api.unpaywall.org/v2/{doi}?email=oa-fetch@chitta-bridge"
                req = urllib.request.Request(api, headers=cls._HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read())
                best = data.get("best_oa_location") or {}
                pdf_url_unpaywall = best.get("url_for_pdf") or best.get("url") or ""
                unpaywall_lines = [
                    f"# {data.get('title', doi)}",
                    f"**Journal:** {data.get('journal_name', '')}  **Year:** {data.get('year', '')}",
                    f"**DOI:** {doi}  **OA status:** {data.get('oa_status', '')}",
                    f"**Authors:** {'; '.join(a.get('family','') + ', ' + a.get('given','') for a in (data.get('z_authors') or [])[:6])}",
                ]
                if pdf_url_unpaywall:
                    unpaywall_lines.append(f"**PDF:** {pdf_url_unpaywall}")
                if data.get("abstract"):
                    unpaywall_lines += ["", "## Abstract", data["abstract"]]
                    return "\n".join(unpaywall_lines)
                # no abstract — fall through to CrossRef which usually has it
            except Exception:
                pass

            # CrossRef — authoritative DOI registry, has abstract + relation/supplement links
            try:
                api = f"https://api.crossref.org/works/{doi}"
                req = urllib.request.Request(
                    api, headers={**cls._HEADERS, "User-Agent": "chitta-bridge/1.0 (mailto:oa-fetch@chitta-bridge)"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read()).get("message", {})
                authors = [
                    f"{a.get('family','')} {a.get('given','')}".strip()
                    for a in (data.get("author") or [])[:6]
                ]
                pub_date = ""
                dp = (data.get("published") or data.get("published-print") or data.get("issued") or {})
                parts_d = dp.get("date-parts", [[]])[0]
                if parts_d:
                    pub_date = "-".join(str(p) for p in parts_d)
                # Use Unpaywall header if available (has PDF, journal), else CrossRef
                if unpaywall_lines:
                    lines = unpaywall_lines
                else:
                    lines = [
                        f"# {' '.join(data.get('title', [doi]))}",
                        f"**Journal:** {data.get('container-title', [''])[0] if data.get('container-title') else ''}  **Year:** {pub_date}",
                        f"**DOI:** {doi}  **Type:** {data.get('type', '')}",
                        f"**Authors:** {'; '.join(authors)}",
                    ]
                # Supplement/related links from CrossRef
                relation = data.get("relation", {})
                for rel_type, items in relation.items():
                    for item in (items if isinstance(items, list) else [items]):
                        lines.append(f"**{rel_type}:** {item.get('id','')} ({item.get('id-type','')})")
                abstract = data.get("abstract") or ""
                if abstract:
                    abstract = re.sub(r"<[^>]+>", " ", abstract)
                    abstract = re.sub(r"\s+", " ", abstract).strip()
                    lines += ["", "## Abstract", abstract]
                return "\n".join(lines)
            except Exception:
                pass

            # OpenAlex fallback
            try:
                api = f"https://api.openalex.org/works/doi:{doi}"
                req = urllib.request.Request(api, headers={**cls._HEADERS, "mailto": "oa-fetch@chitta-bridge"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read())
                oa = data.get("open_access", {})
                authors = [
                    a.get("author", {}).get("display_name", "")
                    for a in (data.get("authorships") or [])[:6]
                ]
                lines = [
                    f"# {data.get('display_name', doi)}",
                    f"**Year:** {data.get('publication_year', '')}",
                    f"**Authors:** {'; '.join(authors)}",
                    f"**DOI:** {doi}",
                ]
                if oa.get("oa_url"):
                    lines.append(f"**PDF:** {oa['oa_url']}")
                abstract = data.get("abstract") or "(abstract not available)"
                lines += ["", "## Abstract", abstract]
                return "\n".join(lines)
            except Exception:
                pass

        return ""

    @classmethod
    def paper_fetch(cls, url_or_doi: str, pdf_path: str = "",
                    full_text: bool = False, timeout: int = 20) -> str:
        """Fetch paper metadata + discover all supplement/data/code resources.

        Strategy (no external services, only stable official APIs):
        1. Paper metadata via bioRxiv/arXiv/DOI APIs
        2. Full text: auto-find local PDF by DOI, or extract if pdf_path given
        3. Supplement discovery via Zenodo, Figshare, GitHub search by DOI
        4. URL extraction from local PDF if available
        """
        # Normalise input to a URL
        url = url_or_doi
        if re.match(r"^10\.\d{4,}/", url_or_doi):
            url = f"https://doi.org/{url_or_doi}"

        # 1. Paper metadata
        meta = cls._academic_fetch(url, timeout=timeout)
        if not meta:
            meta = f"(could not fetch metadata for: {url})"

        # 2. Extract DOI from URL or metadata
        doi = ""
        m = re.search(r"(10\.\d{4,}/[^\s\]\)\"]+)", url + "\n" + meta)
        if m:
            doi = cls._clean_doi(m.group(1).rstrip(".),\"'"))

        supplement_lines: list[str] = []

        # 3. Full text — find local PDF by DOI or use provided pdf_path
        if full_text and not pdf_path and doi:
            # Search common scratch/download locations for a PDF matching the DOI
            doi_stem = doi.split("/")[-1].split("v")[0]  # e.g. "2026.01.22.701213"
            search_dirs = [
                os.path.expanduser("~/Downloads"),
                os.path.expanduser("~/scratch"),
                "/tmp",
                os.environ.get("SCRATCH", ""),
            ]
            # Also try the directory inferred from HOME/scratch patterns
            home = os.environ.get("HOME", "")
            if home:
                search_dirs += [
                    os.path.join(home, "scratch"),
                    os.path.join("/maps/projects/caeg/people", os.environ.get("USER", ""), "scratch"),
                ]
            import glob as _glob
            for sdir in search_dirs:
                if not sdir or not os.path.isdir(sdir):
                    continue
                # depth-limited: check top dir + one level deep (avoids slow NFS walks)
                try:
                    for pattern in (
                        os.path.join(sdir, f"*{doi_stem}*.pdf"),
                        os.path.join(sdir, "*", f"*{doi_stem}*.pdf"),
                    ):
                        matches = _glob.glob(pattern)
                        if matches:
                            pdf_path = matches[0]
                            break
                except Exception:
                    pass
                if pdf_path:
                    break

        if full_text and not pdf_path and doi:
            # Try to download the PDF programmatically before giving up
            _tmp_dir = "/projects/caeg/scratch/kbd606/tmp"
            if not os.path.isdir(_tmp_dir):
                _tmp_dir = os.environ.get("TMPDIR", "/tmp")
            _tmp_pdf = os.path.join(_tmp_dir, doi.split("/")[-1] + ".pdf")

            # Only try Unpaywall OA link — bioRxiv/medRxiv direct PDFs stall on Cloudflare
            _pdf_candidates: list[str] = []
            try:
                _uw_api = f"https://api.unpaywall.org/v2/{doi}?email=chitta@bridge.local"
                _req = urllib.request.Request(_uw_api, headers=cls._HEADERS)
                with urllib.request.urlopen(_req, timeout=8) as _r:
                    _uw = _json.loads(_r.read())
                _best = _uw.get("best_oa_location") or {}
                _oa_url = _best.get("url_for_pdf") or _best.get("url")
                if _oa_url:
                    _pdf_candidates.append(_oa_url)
            except Exception:
                pass

            for _purl in _pdf_candidates:
                try:
                    _req2 = urllib.request.Request(_purl, headers={
                        **cls._HEADERS,
                        "Accept": "application/pdf,*/*",
                    })
                    with urllib.request.urlopen(_req2, timeout=8) as _r2:
                        _content_type = _r2.headers.get("Content-Type", "")
                        _data = _r2.read()
                    if b"%PDF" in _data[:10] or "pdf" in _content_type.lower():
                        with open(_tmp_pdf, "wb") as _fh:
                            _fh.write(_data)
                        pdf_path = _tmp_pdf
                        supplement_lines.append(f"\n(PDF downloaded from {_purl})")
                        break
                except Exception:
                    pass

        if full_text and not pdf_path:
            # PDF not found locally and could not be downloaded
            pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf" if doi else "(unknown)"
            supplement_lines.append(
                f"\n## Full text\n"
                f"PDF is Cloudflare-protected and cannot be downloaded programmatically. "
                f"Download it manually and call:\n"
                f"`paper_fetch(url=\"{url}\", pdf_path=\"/path/to/downloaded.pdf\")`\n"
                f"or use `pdf_read(path=\"/path/to/downloaded.pdf\")` directly.\n"
                f"Direct PDF URL (for browser download): {pdf_url}"
            )
        elif pdf_path:
            supplement_lines.append(f"\n(PDF: {pdf_path})")

        # 4. Scan local PDF — extract full text and/or supplement URLs
        if pdf_path:
            try:
                import pdfplumber
                found_urls: list[str] = []
                full_text_pages: list[str] = []
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        urls = re.findall(r"https?://[^\s\]\)>\"]+", text)
                        found_urls.extend(urls)
                        if full_text:
                            full_text_pages.append(f"\n--- Page {i + 1} ---\n{text.strip()}")
                if full_text and full_text_pages:
                    supplement_lines.append("\n## Full Text")
                    supplement_lines.extend(full_text_pages)
                academic_urls = [
                    u for u in dict.fromkeys(found_urls)
                    if any(k in u.lower() for k in (
                        "zenodo", "figshare", "github", "osf.io", "dryad",
                        "dataverse", "s3.", "data.", "supplement", "code",
                        "gitlab", "bitbucket", "sourceforge",
                    ))
                ]
                if academic_urls:
                    supplement_lines.append("\n## Resources found in PDF")
                    for u in academic_urls[:20]:
                        supplement_lines.append(f"- {u}")
            except Exception as e:
                supplement_lines.append(f"(pdf scan error: {e})")

        # 4. Zenodo search by DOI
        if doi:
            try:
                api = f"https://zenodo.org/api/records?q=related.identifier:{doi}&size=5"
                req = urllib.request.Request(api, headers=cls._HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = _json.loads(resp.read())
                hits = (data.get("hits") or {}).get("hits", [])
                if hits:
                    supplement_lines.append("\n## Zenodo deposits linked to this paper")
                    for hit in hits[:5]:
                        meta_z = hit.get("metadata", {})
                        files = hit.get("files", [])
                        record_id = hit.get("id", "")
                        supplement_lines.append(
                            f"- [{meta_z.get('title','Zenodo')}]"
                            f"(https://zenodo.org/records/{record_id})"
                            f" — {len(files)} file(s), DOI: {meta_z.get('doi','')}"
                        )
            except Exception:
                pass

        # 5. Figshare search by DOI
        if doi:
            try:
                import json as _json2
                api = "https://api.figshare.com/v2/articles/search"
                payload = _json2.dumps({"search_for": doi, "item_type": 3}).encode()
                req = urllib.request.Request(
                    api, data=payload,
                    headers={**cls._HEADERS, "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    results = _json.loads(resp.read())
                if results:
                    supplement_lines.append("\n## Figshare datasets linked to this paper")
                    for r in results[:3]:
                        supplement_lines.append(
                            f"- [{r.get('title','')}]({r.get('url_public_html','')}) "
                            f"DOI: {r.get('doi','')}"
                        )
            except Exception:
                pass

        full = meta
        if supplement_lines:
            full += "\n" + "\n".join(supplement_lines)
        else:
            full += "\n\n(No supplementary resources found via Zenodo/Figshare search)"
        return full
