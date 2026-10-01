"""Entry point: ``python -m helpdesk [config.yaml]``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .app import amain


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="helpdesk", description="SIP voice helpdesk agent (ASR + LLM + TTS)"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.environ.get("HELPDESK_CONFIG", "/app/config/config.yaml"),
        help="path to the YAML configuration file",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(amain(args.config))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
