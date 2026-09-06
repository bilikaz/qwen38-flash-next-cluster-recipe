#!/usr/bin/env bash
# Qwen3.8-Flash-Next (hibrid46, 4.6-bit) on TWO DGX Sparks: TP=2 over the ConnectX link (RDMA).
# First run: no cluster.env → ./setup.sh (finds the second box, the interconnect, opens the firewall). Then:
# pull the image on both boxes, download the weights (once) and sync them to the worker, start the worker
# (--headless) and the head, wait healthy. Everything model-side is recipe.yaml; the boxes are cluster.env.
# OpenAI API on the head at :$PORT. ./stop.sh stops both, ./view.sh shows live stats + the RDMA proof.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source lib.sh
command -v docker >/dev/null || { echo "docker is required"; exit 1; }

# 0. boxes — none configured yet? set the cluster up first.
if ! have_cluster; then
  echo "· no cluster.env — running ./setup.sh first"
  ./setup.sh
fi
load_cluster
ssh_w true 2>/dev/null || { echo "✗ cannot reach the worker $WORKER — rerun ./setup.sh"; exit 1; }

IMAGE="$(rkey server image)";   PORT="$(rkey server port)";   HOST="$(rkey server host)";  HOST="${HOST:-127.0.0.1}"
HF_REPO="$(rkey server model)"; MPORT="$(rkey server master_port)"; MPORT="${MPORT:-25000}"
MODELS_DIR="$(rkey server models_dir)"; CACHE_DIR="$(rkey server cache_dir)"; CPUSET="$(rkey server cpuset)"
mkdir -p "$MODELS_DIR" "$CACHE_DIR"
MODELS_ABS="$(cd "$MODELS_DIR" && pwd)"; CACHE_ABS="$(cd "$CACHE_DIR" && pwd)"
LOCAL_NAME="$(basename "$HF_REPO")"; MODEL_DIR="$MODELS_ABS/$LOCAL_NAME"
ssh_w "mkdir -p '$MODELS_ABS' '$CACHE_ABS'"

# 1. image on BOTH boxes (a public tag — each box pulls; no-op when present)
echo "· image $IMAGE — head"; docker pull -q "$IMAGE" >/dev/null || docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "✗ cannot pull $IMAGE"; exit 1; }
echo "· image $IMAGE — worker"; ssh_w "docker pull -q '$IMAGE' >/dev/null || docker image inspect '$IMAGE' >/dev/null 2>&1" || { echo "✗ worker cannot pull $IMAGE"; exit 1; }

# 2. weights: ~91G, resumable — download on the head, then sync to the worker at the SAME path
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "· downloading $HF_REPO -> $MODEL_DIR"
  if command -v hf >/dev/null; then
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  else
    TTY=""; [ -t 1 ] && TTY="-t"
    docker run --rm $TTY -e HF_TOKEN -v "$MODELS_ABS:/dl" --entrypoint python3 "$IMAGE" \
      -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_REPO', local_dir='/dl/$LOCAL_NAME')"
  fi
fi
if ! ssh_w "[ -f '$MODEL_DIR/model.safetensors.index.json' ]"; then
  echo "· syncing weights to the worker (one-time, ~91G over ssh — resumable, rerun if interrupted)"
  rsync -a --size-only --info=progress2 -e "ssh -o BatchMode=yes" "$MODEL_DIR/" "$WORKER:$MODEL_DIR/"
fi
echo "✓ weights on both boxes: $MODEL_DIR"
# The container mounts the kit's models/ folder at /models, nothing else — a symlink for models/<name> would dangle
# inside the container (its target is outside the mount) and vLLM would mistake the path for a Hugging Face repo id.
# So models/<name> must be a REAL directory: a download, or a hardlink copy (`cp -al /path/to/checkpoint models/<name>`,
# instant, zero extra space, same filesystem) — never a symlink.
for side in head worker; do
  if [ "$side" = head ]; then islink=$([ -L "$MODEL_DIR" ] && echo yes || echo no); else islink=$(ssh_w "[ -L '$MODEL_DIR' ] && echo yes || echo no"); fi
  if [ "$islink" = yes ]; then
    echo "✗ $side: $MODEL_DIR is a symlink — the container cannot follow it. Replace it with a real directory:"
    echo "    rm $MODEL_DIR && cp -al /path/to/Qwen3.8-Flash-Next-hibrid46 $MODEL_DIR     # hardlink copy, instant"
    exit 1
  fi
done


# 4. compose the docker run for a rank. Host networking + the three RDMA flags (without them NCCL silently
#    falls back to TCP over the same cable — half the speed, no error). Per-box pins: NCCL/gloo on the
#    interconnect iface, VLLM_HOST_IP = that box's interconnect IP (the LAN must never carry cluster traffic).
ENVS=(); while IFS=$'\t' read -r k v; do [ -n "$k" ] && ENVS+=(-e "$k=$v"); done < <(rsection env)
FLAGS=(); while IFS=$'\t' read -r k v; do
  case "$v" in true) FLAGS+=("--$k");; false|null|"") ;; *) FLAGS+=("--$k" "$v");; esac
done < <(rsection vllm)
compose() {  # compose <rank> <iface> <ic-ip> <hca> <has-rdma yes|no>  → prints the docker run command (quoted)
  local rank=$1 iface=$2 ic=$3 hca=$4 rdma=$5 a=()
  a=(docker run -d --name "$NAME" --gpus all --ipc=host --network host --cap-add SYS_PTRACE)
  [ "$rdma" = yes ] && a+=(--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1)
  [ -n "$CPUSET" ] && a+=(--cpuset-cpus "$CPUSET")
  a+=(-v "$MODELS_ABS:/models" -v "$CACHE_ABS:/cache"
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace -e VLLM_CACHE_ROOT=/cache/vllm-cache
      -e "NCCL_SOCKET_IFNAME=$iface" -e "GLOO_SOCKET_IFNAME=$iface" -e "VLLM_HOST_IP=$ic" -e NCCL_IB_DISABLE=0)
  [ -n "$hca" ] && a+=(-e "NCCL_IB_HCA=$hca")
  a+=("${ENVS[@]}" --entrypoint vllm "$IMAGE" serve "/models/$LOCAL_NAME" --host "$HOST" --port "$PORT"
      --nnodes 2 --node-rank "$rank" --master-addr "$HEAD_IC" --master-port "$MPORT" --tensor-parallel-size 2)
  [ "$rank" != 0 ] && a+=(--headless)
  a+=("${FLAGS[@]}")
  printf '%q ' "${a[@]}"
}
HEAD_RDMA=$([ -d /dev/infiniband ] && echo yes || echo no)
WORKER_RDMA=$(ssh_w "[ -d /dev/infiniband ] && echo yes || echo no")
[ "$HEAD_RDMA$WORKER_RDMA" = yesyes ] || echo "  ⚠ RDMA not available on both boxes (head $HEAD_RDMA, worker $WORKER_RDMA) — running NCCL over TCP"
compaction_check

# 5. launch: clear old containers, then GATE on memory (unified memory needs ~30-60 s after a container dies;
#    launching earlier = a phantom CUDA OOM), then HEAD first (the rendezvous master), then the worker — the order
#    every successful TP=2 boot of this model used; the worker retries the connect until the head listens.
docker rm -f "$NAME" >/dev/null 2>&1 || true
ssh_w "docker rm -f '$NAME' >/dev/null 2>&1 || true"
wait_mem 100 120 || exit 1
evict_cache "$MODEL_DIR"
echo "· starting head (rank 0) — API on $HOST:$PORT once healthy (first boot ~10 min: load + compile warmup)"
eval "$(compose 0 "$HEAD_IFACE" "$HEAD_IC" "$HEAD_HCA" "$HEAD_RDMA")" >/dev/null
sleep 2
echo "· starting worker ($WORKER_HOST, rank 1, --headless)"
ssh_w "$(compose 1 "$WORKER_IFACE" "$WORKER_IC" "$WORKER_HCA" "$WORKER_RDMA") >/dev/null"

# 6. stream the head's logs until healthy; fail fast if either container dies
echo "· streaming engine logs until healthy (Ctrl-C detaches; the cluster keeps booting)"
docker logs -f "$NAME" 2>&1 &
LOGS=$!
trap 'kill "$LOGS" 2>/dev/null' EXIT INT TERM
for i in $(seq 1 360); do
  if curl -sf -m 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    kill "$LOGS" 2>/dev/null; wait "$LOGS" 2>/dev/null
    echo
    echo "──────────────────────────────────────────────────────────"
    echo "✓ cluster serving — OpenAI-compatible API is live on the head"
    echo "    endpoint : http://$HOST:$PORT/v1"
    echo "    monitor  : ./view.sh          (throughput, acceptance, RDMA proof)"
    echo "    logs     : docker logs -f $NAME      · worker: ssh $WORKER docker logs -f $NAME"
    echo "    stop     : ./stop.sh          (both boxes)"
    echo "──────────────────────────────────────────────────────────"
    exit 0
  fi
  if ! docker ps -q --filter "name=^$NAME\$" | grep -q .; then
    kill "$LOGS" 2>/dev/null; wait "$LOGS" 2>/dev/null
    echo "✗ head container exited — see above. Worker's last lines:"; ssh_w "docker logs --tail 20 '$NAME'" 2>&1 | tail -20; exit 1
  fi
  if ! ssh_w "docker ps -q --filter 'name=^$NAME\$' | grep -q ." 2>/dev/null; then
    kill "$LOGS" 2>/dev/null; wait "$LOGS" 2>/dev/null
    echo "✗ worker container exited:"; ssh_w "docker logs --tail 40 '$NAME'" 2>&1 | tail -40
    docker rm -f "$NAME" >/dev/null 2>&1; exit 1
  fi
  sleep 5
done
echo "✗ not healthy after 30 min — still booting? watch: docker logs -f $NAME"; exit 1
