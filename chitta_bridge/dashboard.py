"""Standalone rooms dashboard — run independently of the MCP bridge."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from chitta_bridge.server import _start_dashboard


async def main(port: int = 7680):
    await _start_dashboard(port)
    print(f"Dashboard running at http://localhost:{port}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7680
    asyncio.run(main(port))
