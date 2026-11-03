#!/usr/bin/env bash
set -euo pipefail

# Not "python -m pytest": that would put the source tree on sys.path
# and hide packaging bugs in the installed package.
pytest -q test
