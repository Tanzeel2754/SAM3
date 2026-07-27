#!/usr/bin/env python
"""Start the SAM3 web application."""

import argparse
import os
import socket
import sys
from pathlib import Path

# Repo root (sam3/) must be on PYTHONPATH so `webapp` is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))

import uvicorn

from webapp.backend.app import app


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def main():
    parser = argparse.ArgumentParser(description="SAM3 Studio web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    if _port_in_use(args.host, args.port):
        print(
            f"\nError: port {args.port} is already in use on {args.host}.\n"
            f"  • Stop the other server, or run on a different port:\n"
            f"      python webapp/run.py --port {args.port + 1}\n"
            f"  • On Windows, find and kill the process:\n"
            f"      netstat -ano | findstr :{args.port}\n"
            f"      taskkill /PID <pid> /F\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.reload:
        uvicorn.run(
            "webapp.backend.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(REPO_ROOT / "webapp")],
            log_level="info",
        )
    else:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )


if __name__ == "__main__":
    main()
