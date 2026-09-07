#!/usr/bin/env bash
# Live view: both containers' status, API health, the RDMA PROOF (interconnect counters moving while the TCP path
# stays flat), then the head's engine log (throughput windows, KV usage, speculative-decoding acceptance).
# Ctrl-C detaches; the cluster keeps running.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source lib.sh
load_cluster || { echo "no cluster.env — ./run.sh first"; exit 1; }
PORT="$(rkey server port)"; HOST="$(rkey server host)"; HOST="${HOST:-127.0.0.1}"

SH="$(docker ps --filter "name=^$NAME\$" --format '{{.Status}}')"
SW="$(ssh_w "docker ps --filter 'name=^$NAME\$' --format '{{.Status}}'" 2>/dev/null || true)"
[ -n "$SH" ] || { echo "head: not running (./run.sh starts the cluster)"; exit 1; }
H="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://$HOST:${PORT:-8000}/health" || true)"
echo "· head   ($HEAD_IC via $HEAD_IFACE${HEAD_HCA:+, $HEAD_HCA}): $SH — health HTTP $H — API http://$HOST:${PORT:-8000}/v1"
echo "· worker ($WORKER_IC via $WORKER_IFACE${WORKER_HCA:+, $WORKER_HCA}): ${SW:-NOT RUNNING}"

# RDMA proof: 3-second sample. Verbs traffic shows on the HCA's port counter (×4 = bytes); TCP traffic would show
# on the netdev tx counter instead. During decode expect tens of MB on the RDMA line and ~0 on the netdev line.
H1="${HEAD_HCA%%,*}"   # first device of a comma list (both halves carry traffic; one counter proves RDMA)
if [ -n "$H1" ] && [ -d "/sys/class/infiniband/$H1" ]; then
  C="/sys/class/infiniband/$H1/ports/1/counters/port_xmit_data"; T="/sys/class/net/$HEAD_IFACE/statistics/tx_bytes"
  a=$(cat "$C"); t=$(cat "$T"); sleep 3; b=$(cat "$C"); u=$(cat "$T")
  echo "· transport (3 s): RDMA $(( (b-a)*4/1048576 )) MB · TCP $(( (u-t)/1048576 )) MB  → $([ $(( (b-a)*4 )) -gt $(( u-t )) ] && echo 'NCCL on RDMA ✓' || echo 'idle, or NCCL on TCP — send a request and look again')"
else
  echo "· transport: no RDMA device on this link — NCCL runs over TCP"
fi
echo "  (Ctrl-C detaches)"
exec docker logs -f --tail 30 "$NAME"
