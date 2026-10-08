"""Logging setup shared by the command-line entry points."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Plain messages on stdout, so `| tee` and the preflight checks see them."""
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout, force=True)
