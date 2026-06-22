"""Persistent JSONL reference library with git auto-commit."""

import os
import re
import json
import datetime
import subprocess
import urllib.request
import xml.etree.ElementTree as ET

__all__ = ["RefLib"]

_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")


def _clean_doi(doi: str) -> str:
    doi = doi.strip()
    if not _DOI_RE.match(doi) or "/.." in doi or "../" in doi:
        return ""
    return doi


class RefLib:
    """Persistent JSONL reference library. One JSON object per line, DOI as dedup key.

    Storage: $CHITTA_REFLIB or ~/.chitta/reflib.jsonl
    Git auto-commit on add/remove/tag when the file is inside a git repo.
    """

    _DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".chitta", "reflib.jsonl")

    @classmethod
    def _path(cls) -> str:
        return os.environ.get("CHITTA_REFLIB", cls._DEFAULT_PATH)

    @classmethod
    def _load(cls) -> list:
        p = cls._path()
        if not os.path.exists(p):
            return []
        entries = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        return entries

    @classmethod
    def _save(cls, entries: list) -> None:
        p = cls._path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @classmethod
    def _git_commit(cls, msg: str) -> None:
        p = cls._path()
        d = os.path.dirname(p)
        try:
            subprocess.run(["git", "-C", d, "add", os.path.basename(p)],
                           capture_output=True, timeout=5)
            subprocess.run(["git", "-C", d, "commit", "-m", msg],
                           capture_output=True, timeout=5)
        except Exception:
            pass

    @classmethod
    def _key(cls, entry: dict) -> str:
        return (entry.get("doi") or entry.get("url") or "").lower().rstrip("/")

    @classmethod
    def _fetch_meta(cls, url_or_doi: str) -> dict:
        """Return structured metadata dict from a DOI or URL using open APIs only."""
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
                    data = json.loads(resp.read())
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
                    data = json.loads(resp.read()).get("message", {})
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
                    data = json.loads(resp.read())
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

    @classmethod
    def add(cls, url_or_doi: str, tags: list | None = None, notes: str = "") -> str:
        """Add one or more papers (space/comma/newline-separated DOIs or URLs)."""
        entries = cls._load()
        keys = {cls._key(e) for e in entries}
        added, dupes, failed = [], [], []

        items = [x.strip() for x in re.split(r"[\s,]+", url_or_doi) if x.strip()]
        for item in items:
            meta = cls._fetch_meta(item)
            if not meta or not meta.get("title"):
                failed.append(item)
                continue
            k = (meta.get("doi") or "").lower()
            if k and k in keys:
                dupes.append(meta["title"])
                continue
            entry = {
                **meta,
                "tags": list(tags or []),
                "notes": notes,
                "added": datetime.datetime.utcnow().isoformat()[:19] + "Z",
            }
            entries.append(entry)
            keys.add(k)
            added.append(entry)

        if added:
            cls._save(entries)
            cls._git_commit(f"reflib: add {len(added)} paper(s)")

        lines = []
        for e in added:
            lines.append(f"+ [{e.get('doi','')}] {e.get('title','')} ({e.get('year','')})")
        for d in dupes:
            lines.append(f"= already in library: {d}")
        for f in failed:
            lines.append(f"✗ could not fetch metadata: {f}")
        lines.append(f"\nLibrary: {len(entries)} papers total  ({cls._path()})")
        return "\n".join(lines)

    @classmethod
    def search(cls, query: str = "", tag: str = "", limit: int = 20) -> str:
        entries = cls._load()
        if not entries:
            return "Reference library is empty. Use reflib_add to add papers."
        kw = [k.lower() for k in query.split() if k.strip()] if query else []
        results = []
        for e in entries:
            if tag and tag.lower() not in [t.lower() for t in (e.get("tags") or [])]:
                continue
            if kw:
                text = " ".join([
                    e.get("title", ""),
                    e.get("abstract", ""),
                    " ".join(e.get("authors") or []),
                    e.get("notes", ""),
                    " ".join(e.get("tags") or []),
                ]).lower()
                if not all(k in text for k in kw):
                    continue
            results.append(e)
        results = results[:limit]
        if not results:
            suffix = f" with tag {tag!r}" if tag else ""
            return f"No results for {query!r}{suffix}  (library has {len(entries)} papers)"
        lines = [f"Found {len(results)} paper(s) (of {len(entries)} total):\n"]
        for e in results:
            tags_str = f"\n  Tags: {', '.join(e.get('tags',[]))}" if e.get("tags") else ""
            notes_str = f"\n  Notes: {e['notes'][:120]}" if e.get("notes") else ""
            lines.append(
                f"[{e.get('doi') or e.get('url','')}] {e.get('title','')} ({e.get('year','')})\n"
                f"  {', '.join((e.get('authors') or [])[:4])}  ·  {e.get('journal','')}"
                + tags_str + notes_str
            )
        return "\n\n".join(lines)

    @classmethod
    def remove(cls, doi_or_title: str) -> str:
        entries = cls._load()
        query = doi_or_title.lower().strip()
        keep, removed = [], []
        for e in entries:
            if (e.get("doi") or "").lower() == query or query in (e.get("title") or "").lower():
                removed.append(e.get("title") or e.get("doi") or "")
            else:
                keep.append(e)
        if not removed:
            return f"No entry matching {doi_or_title!r}"
        cls._save(keep)
        cls._git_commit(f"reflib: remove {len(removed)} paper(s)")
        return "\n".join(f"- removed: {r}" for r in removed)

    @classmethod
    def tag(cls, doi_or_title: str, tags: list, notes: str = "", replace: bool = False) -> str:
        entries = cls._load()
        query = doi_or_title.lower().strip()
        updated = []
        for e in entries:
            if (e.get("doi") or "").lower() == query or query in (e.get("title") or "").lower():
                if replace:
                    e["tags"] = list(tags)
                else:
                    existing = e.get("tags") or []
                    e["tags"] = existing + [t for t in tags if t not in existing]
                if notes:
                    e["notes"] = notes
                updated.append(e.get("title") or e.get("doi") or "")
        if not updated:
            return f"No entry matching {doi_or_title!r}"
        cls._save(entries)
        cls._git_commit(f"reflib: tag {len(updated)} paper(s)")
        return "\n".join(f"tagged: {u}" for u in updated)

    @classmethod
    def export(cls, fmt: str = "markdown", tag: str = "", query: str = "") -> str:
        entries = cls._load()
        if not entries:
            return "Reference library is empty."
        if tag or query:
            kw = [k.lower() for k in query.split() if k.strip()] if query else []
            filtered = []
            for e in entries:
                if tag and tag.lower() not in [t.lower() for t in (e.get("tags") or [])]:
                    continue
                if kw:
                    text = " ".join([e.get("title",""), e.get("abstract",""),
                                     " ".join(e.get("authors") or [])]).lower()
                    if not all(k in text for k in kw):
                        continue
                filtered.append(e)
            entries = filtered

        if fmt == "bibtex":
            lines = []
            for e in entries:
                doi = e.get("doi", "")
                family = ((e.get("authors") or ["anon"])[0]).split()[-1]
                key = re.sub(r"[^a-zA-Z0-9]", "", family) + e.get("year", "")
                lines.append(
                    f"@article{{{key},\n"
                    f"  title   = {{{e.get('title','')}}},\n"
                    f"  author  = {{{' and '.join(e.get('authors') or [])}}},\n"
                    f"  year    = {{{e.get('year','')}}},\n"
                    f"  journal = {{{e.get('journal','')}}},\n"
                    f"  doi     = {{{doi}}},\n"
                    f"}}"
                )
            return "\n\n".join(lines)

        if fmt == "jsonl":
            return "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)

        # markdown (default)
        lines = [f"# Reference Library ({len(entries)} papers)\n"]
        by_year: dict = {}
        for e in entries:
            by_year.setdefault(e.get("year", "?"), []).append(e)
        for yr in sorted(by_year.keys(), reverse=True):
            lines.append(f"\n## {yr}")
            for e in by_year[yr]:
                doi = e.get("doi", "")
                auths = list(e.get("authors") or [])
                auth_str = ", ".join(auths[:3]) + (" et al." if len(auths) > 3 else "")
                tags_str = f" `{'` `'.join(e.get('tags',[]))}`" if e.get("tags") else ""
                lines.append(
                    f"- **{e.get('title','')}** — {auth_str}. *{e.get('journal','')}*{tags_str}  \n"
                    f"  DOI: [{doi}](https://doi.org/{doi})"
                )
                if e.get("notes"):
                    lines.append(f"  > {e['notes']}")
        return "\n".join(lines)
