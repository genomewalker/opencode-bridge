---
name: bridge-install-browser
description: Install the optional Cloudflare-bypass stack (curl_cffi + playwright 1.49 + camoufox) for the opencode-bridge browser_fetch tool, then download the camoufox Firefox binary and verify. Run this when browser_fetch falls back to web_fetch because the stack is absent, or after a fresh `uv sync` that dropped the extra.
---

# Install the browser_fetch stack

`browser_fetch` works without these deps by falling back to the plain
`web_fetch` route. To enable the full Cloudflare-managed-challenge bypass
(curl_cffi Firefox-JA3 first, camoufox mints a `cf_clearance` cookie only on
challenge), install the `browser` extra and fetch the camoufox binary.

Run from the opencode-bridge repo root.

## Steps

1. Install the extra into the project venv:
   ```bash
   uv sync --extra browser
   ```

2. Download the fortified-Firefox binary (~700 MB, not pulled by pip):
   ```bash
   .venv/bin/camoufox fetch
   ```

3. Verify both layers are present:
   ```bash
   .venv/bin/python -c "from chitta_bridge.search.browser import BrowserFetch as B; print('curl_cffi', B.available(), '| camoufox', B.can_mint())"
   ```
   Expect `curl_cffi True | camoufox True`.

4. Smoke-test the lightweight path (must return text, never touch the browser):
   ```bash
   .venv/bin/python -c "from chitta_bridge.search.browser import BrowserFetch as B; print(B.fetch('https://example.com', max_chars=120)[:120]); print('browser used:', bool(B._cache))"
   ```
   Expect the Example-Domain text and `browser used: False`.

## Notes

- Restart the bridge after install so the new deps load (lazy imports mean a
  running bridge won't pick them up until reimport).
- `camoufox fetch` is idempotent; skip step 2 if `~/.cache/camoufox` already
  holds the `135.x` binary.
- To remove the stack later: `uv sync` (without `--extra browser`).
  `browser_fetch` then silently reverts to the `web_fetch` fallback.
