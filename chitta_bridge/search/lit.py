"""Literature search wrappers — no auth required except OpenAlex."""

from __future__ import annotations

import re

__all__ = ["LitSearch"]

_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")


def _clean_doi(doi: str) -> str:
    """Validate a DOI for safe interpolation into API URLs ('' if unsafe)."""
    doi = doi.strip()
    if not _DOI_RE.match(doi) or "/.." in doi or "../" in doi:
        return ""
    return doi


class LitSearch:
    """Thin wrappers around public literature APIs — no auth required except OpenAlex."""

    _RATE = 1.0  # seconds between requests (conservative)

    @staticmethod
    def _get(url: str, params: dict | None = None, timeout: int = 15) -> dict | str:
        import urllib.request
        import urllib.parse
        import json as _json
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "chitta-bridge/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
        try:
            return _json.loads(body)
        except Exception:
            return body

    @classmethod
    def arxiv(cls, query: str, max_results: int = 10,
               sort_by: str = "relevance") -> str:
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom",
              "arxiv": "http://arxiv.org/schemas/atom"}
        params = {"search_query": query, "max_results": max_results,
                  "sortBy": sort_by, "sortOrder": "descending"}
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "chitta-bridge/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
        root = ET.fromstring(body)
        entries = root.findall("atom:entry", ns)
        if not entries:
            return f"No arXiv results for: {query}"
        lines = [f"arXiv search: {query!r} — {len(entries)} results\n"]
        for e in entries:
            title = (e.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            arxiv_id = (e.findtext("atom:id", "", ns) or "").split("/abs/")[-1]
            published = (e.findtext("atom:published", "", ns) or "")[:10]
            summary = (e.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")[:300]
            authors = [a.findtext("atom:name", "", ns) for a in e.findall("atom:author", ns)][:4]
            lines.append(
                f"[{arxiv_id}] {title}\n"
                f"  Authors: {', '.join(authors)}\n"
                f"  Published: {published}\n"
                f"  Abstract: {summary}...\n"
                f"  URL: https://arxiv.org/abs/{arxiv_id}\n"
            )
        return "\n".join(lines)

    @classmethod
    def biorxiv(cls, query: str, start_date: str, end_date: str,
                server: str = "biorxiv", max_results: int = 20) -> str:
        import time
        if server not in ("biorxiv", "medrxiv"):
            return f"bioRxiv API error: invalid server '{server}' (use biorxiv or medrxiv)"
        for d in (start_date, end_date):
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                return f"bioRxiv API error: invalid date '{d}' (use YYYY-MM-DD)"
        results = []
        cursor = 0
        while len(results) < max_results:
            url = f"https://api.biorxiv.org/details/{server}/{start_date}/{end_date}/{cursor}"
            data = cls._get(url)
            if not isinstance(data, dict):
                return f"bioRxiv API error: {str(data)[:200]}"
            collection = data.get("collection", [])
            if not collection:
                break
            kw = query.lower().split()
            for item in collection:
                text = f"{item.get('title','')} {item.get('abstract','')}".lower()
                if all(k in text for k in kw):
                    results.append(item)
                if len(results) >= max_results:
                    break
            if len(collection) < 100:
                break
            cursor += 100
            time.sleep(cls._RATE)
        if not results:
            return f"No {server} results for {query!r} between {start_date} and {end_date}"
        lines = [f"{server} search: {query!r} ({start_date}→{end_date}) — {len(results)} results\n"]
        for r in results:
            doi = r.get("doi", "")
            lines.append(
                f"[{doi}] {r.get('title','').strip()}\n"
                f"  Authors: {r.get('authors','')[:120]}\n"
                f"  Date: {r.get('date','')}\n"
                f"  Abstract: {r.get('abstract','').strip()[:300]}...\n"
                f"  URL: https://doi.org/{doi}\n"
            )
        return "\n".join(lines)

    @classmethod
    def europepmc(cls, query: str, max_results: int = 20,
                  open_access_only: bool = True) -> str:
        full_query = query + (" AND OPEN_ACCESS:y" if open_access_only else "")
        params = {"query": full_query, "resultType": "lite",
                  "pageSize": min(max_results, 100), "format": "json"}
        data = cls._get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params)
        if not isinstance(data, dict):
            return f"Europe PMC error: {str(data)[:200]}"
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return f"No Europe PMC results for: {query}"
        lines = [f"Europe PMC search: {query!r} — {len(results)} results "
                 f"({'open access only' if open_access_only else 'all'})\n"]
        for r in results:
            pmid = r.get("pmid", r.get("pmcid", ""))
            lines.append(
                f"[{pmid}] {r.get('title','').strip()}\n"
                f"  Authors: {r.get('authorString','')[:120]}\n"
                f"  Journal: {r.get('journalTitle','')}  {r.get('pubYear','')}\n"
                f"  DOI: {r.get('doi','')}\n"
                f"  URL: https://europepmc.org/article/{r.get('source','MED')}/{pmid}\n"
            )
        return "\n".join(lines)

    @classmethod
    def openalex(cls, query: str, entity_type: str = "works",
                 max_results: int = 20, filters: str = "") -> str:
        import os
        api_key = os.environ.get("OPENALEX_API_KEY", "")
        params: dict = {"search": query, "per-page": min(max_results, 100)}
        if filters:
            params["filter"] = filters
        if api_key:
            params["api_key"] = api_key
        else:
            params["mailto"] = "chitta-bridge@localhost"  # polite pool
        data = cls._get(f"https://api.openalex.org/{entity_type}", params)
        if not isinstance(data, dict):
            return f"OpenAlex error: {str(data)[:200]}"
        results = data.get("results", [])
        meta = data.get("meta", {})
        if not results:
            return f"No OpenAlex results for: {query}"
        lines = [f"OpenAlex search: {query!r} — {meta.get('count', len(results))} total, "
                 f"showing {len(results)}\n"]
        for r in results:
            oa_id = r.get("id", "").replace("https://openalex.org/", "")
            title = r.get("display_name", r.get("title", "")).strip()
            year = r.get("publication_year", "")
            doi = r.get("doi", "")
            authors = [a.get("author", {}).get("display_name", "")
                       for a in r.get("authorships", [])[:4]]
            cited = r.get("cited_by_count", "")
            lines.append(
                f"[{oa_id}] {title}\n"
                f"  Authors: {', '.join(authors)}\n"
                f"  Year: {year}  Cited by: {cited}\n"
                f"  DOI: {doi}\n"
            )
        return "\n".join(lines)

    @classmethod
    def _fetch_meta(cls, url_or_doi: str) -> dict:
        """Return structured metadata dict from a DOI or URL using open APIs only."""
        import urllib.request
        import json as _json

        _headers = {"User-Agent": "chitta-bridge/1.0 (mailto:oa-fetch@chitta-bridge)"}

        url = url_or_doi.strip()
        if re.match(r"^10\.\d{4,}/", url):
            url = f"https://doi.org/{url}"

        doi = ""
        m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", url)
        if m:
            doi = _clean_doi(m.group(1).rstrip(".,)\"'"))

        # ── bioRxiv / medRxiv ────────────────────────────────────────────
        mb = re.match(
            r"https?://(?:www\.)?(biorxiv|medrxiv)\.org/content/([^?\s]+?)(?:v\d+)?(?:\.full)?",
            url,
        )
        servers = [mb.group(1)] if mb else ["biorxiv", "medrxiv"]
        _doi_try = (_clean_doi(mb.group(2)) if mb else doi) if servers else doi
        for _srv in servers:
            if not _doi_try:
                break
            try:
                api = f"https://api.biorxiv.org/details/{_srv}/{_doi_try}/na/json"
                req = urllib.request.Request(api, headers=_headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = _json.loads(resp.read())
                items = data.get("collection", [])
                if items:
                    p = items[-1]
                    if p.get("title"):
                        return {
                            "doi": p.get("doi", _doi_try),
                            "url": f"https://doi.org/{p.get('doi', _doi_try)}",
                            "title": p.get("title", "").strip(),
                            "authors": [a.strip() for a in p.get("authors", "").split(";") if a.strip()],
                            "year": (p.get("date") or "")[:4],
                            "journal": _srv,
                            "abstract": (p.get("abstract") or "").strip(),
                            "source": _srv,
                        }
            except Exception:
                pass

        # ── arXiv ────────────────────────────────────────────────────────
        ma = re.match(r"https?://arxiv\.org/(?:abs|pdf)/(\S+?)(?:v\d+)?(?:\.pdf)?/?$", url)
        if ma:
            arxiv_id = ma.group(1)
            try:
                import xml.etree.ElementTree as ET
                api = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
                req = urllib.request.Request(api, headers=_headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    body = resp.read().decode()
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(body)
                entry = root.find("atom:entry", ns)
                if entry is not None:
                    title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
                    authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
                    published = (entry.findtext("atom:published", "", ns) or "")[:10]
                    summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
                    adoi = ""
                    for lnk in entry.findall("atom:link", ns):
                        href = lnk.get("href", "")
                        if "doi.org" in href:
                            dm = re.search(r"(10\.\d{4,}/\S+)", href)
                            if dm:
                                adoi = _clean_doi(dm.group(1))
                    return {
                        "doi": adoi or arxiv_id,
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "title": title,
                        "authors": authors,
                        "year": published[:4],
                        "journal": "arXiv",
                        "abstract": summary,
                        "source": "arxiv",
                    }
            except Exception:
                pass

        # ── CrossRef (any DOI) ───────────────────────────────────────────
        if doi:
            try:
                api = f"https://api.crossref.org/works/{doi}"
                req = urllib.request.Request(api, headers=_headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = _json.loads(resp.read()).get("message", {})
                authors = [
                    f"{a.get('family','')} {a.get('given','')}".strip()
                    for a in (data.get("author") or [])[:10]
                ]
                dp = (data.get("published") or data.get("published-print") or data.get("issued") or {})
                parts = dp.get("date-parts", [[]])[0]
                year = str(parts[0]) if parts else ""
                abstract = data.get("abstract", "")
                if abstract:
                    abstract = re.sub(r"<[^>]+>", " ", abstract)
                    abstract = re.sub(r"\s+", " ", abstract).strip()
                journal = (data.get("container-title") or [""])[0]
                return {
                    "doi": doi,
                    "url": f"https://doi.org/{doi}",
                    "title": " ".join(data.get("title", [doi])),
                    "authors": authors,
                    "year": year,
                    "journal": journal,
                    "abstract": abstract,
                    "source": "crossref",
                }
            except Exception:
                pass

            # ── OpenAlex fallback ─────────────────────────────────────────
            try:
                api = f"https://api.openalex.org/works/doi:{doi}"
                req = urllib.request.Request(api, headers={**_headers, "mailto": "oa-fetch@chitta-bridge"})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = _json.loads(resp.read())
                authors = [
                    a.get("author", {}).get("display_name", "")
                    for a in (data.get("authorships") or [])[:10]
                ]
                journal = ((data.get("primary_location") or {}).get("source") or {}).get("display_name", "")
                return {
                    "doi": doi,
                    "url": f"https://doi.org/{doi}",
                    "title": data.get("display_name", doi),
                    "authors": authors,
                    "year": str(data.get("publication_year", "")),
                    "journal": journal,
                    "abstract": data.get("abstract") or "",
                    "source": "openalex",
                }
            except Exception:
                pass

        return {}
