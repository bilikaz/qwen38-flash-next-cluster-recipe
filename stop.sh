#!/usr/bin/env bash
# Stop and remove the serve container on BOTH boxes. Weights and caches stay — ./run.sh brings it back fast.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source lib.sh
docker rm -f "$NAME" >/dev/null 2>&1 && echo "✓ head stopped" || echo "· head: not running"
if load_cluster 2>/dev/null; then
  ssh_w "docker rm -f '$NAME' >/dev/null 2>&1" && echo "✓ worker stopped ($WORKER_HOST)" || echo "· worker: not running"
else
  echo "· no cluster.env — worker untouched"
fi
