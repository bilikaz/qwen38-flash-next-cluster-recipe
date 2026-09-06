# Shared helpers for setup.sh / run.sh / stop.sh / view.sh. Sourced; the scripts cd to the kit dir first.
# Head = the machine you run these on (serves the API). Worker = the second Spark, reached over ssh.

NAME="qwen38-flash-next-cluster"          # container name on BOTH boxes
CLUSTER_ENV="cluster.env"                  # written by setup.sh: HEAD_*/WORKER_* (machine-specific, gitignored)

# --- tiny recipe.yaml reader (two-level: section -> key: value; strips quotes/comments) ---------
rkey() {  # rkey <section> <key>
  awk -v s="$1" -v k="$2" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec) }
    sec==s && $1==k":" {
      sub(/^[ ]*[^:]*:[ ]*/,""); sub(/[ ]+#.*$/,"")
      gsub(/^["\x27]|["\x27]$/,""); print; exit
    }' recipe.yaml
}
rsection() {  # all key/value lines of a section, "key<TAB>value" (quotes/comments stripped)
  awk -v s="$1" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec); next }
    sec==s && $1 ~ /^[A-Za-z0-9_-]+:$/ || (sec==s && /^[ ]+[A-Za-z0-9_-]+:[ ]/) {
      line=$0; sub(/^[ ]+/,"",line)
      key=line; sub(/:.*/,"",key)
      val=line; sub(/^[^:]*:[ ]*/,"",val); sub(/[ ]+#.*$/,"",val)
      gsub(/^["\x27]|["\x27]$/,"",val)
      if (key != "") print key "\t" val
    }' recipe.yaml
}

# --- cluster.env ----------------------------------------------------------------------------------
have_cluster() { [ -f "$CLUSTER_ENV" ]; }
load_cluster() {
  have_cluster || return 1
  # shellcheck disable=SC1090
  source "$CLUSTER_ENV"
  : "${WORKER_HOST:?cluster.env is incomplete — rerun ./setup.sh}"
  WORKER="${WORKER_USER:+$WORKER_USER@}$WORKER_HOST"
}
ssh_w()  { ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER" "$@"; }         # run on the worker
ssh_wt() { ssh -t -o ConnectTimeout=8 "$WORKER" "$@"; }                         # …with a tty (sudo may prompt)

# --- probing ------------------------------------------------------------------------------------------
# Emits, for a box: "IFACE <name> <ip> <hca|->" per global IPv4 interface (RDMA HCA bound to it, if any),
# "GPU …", "DOCKER …", "NVRT yes|no" (nvidia container runtime), "RDMA yes|no" (/dev/infiniband present),
# "MEM <free GiB>". Runs locally (no arg) or over ssh (<user@host>).
PROBE='
  ip -o -4 addr show 2>/dev/null | while read -r _ ifc _ cidr _; do
    case "$ifc" in lo|docker*|veth*|br-*|virbr*|cni*|flannel*|tailscale*|wg*) continue;; esac
    hca=$(ls /sys/class/net/$ifc/device/infiniband/ 2>/dev/null | head -1)
    echo "IFACE $ifc ${cidr%%/*} ${hca:--}"
  done
  echo "GPU $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
  echo "DOCKER $(docker --version 2>/dev/null || echo none)"
  echo "NVRT $(docker info 2>/dev/null | grep -qi nvidia && echo yes || echo no)"
  echo "RDMA $([ -d /dev/infiniband ] && echo yes || echo no)"
  echo "MEM $(free -g 2>/dev/null | awk "/^Mem:/{print \$7}")"'
probe() {  # probe [user@host]
  if [ -n "${1:-}" ]; then ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" "$PROBE"; else bash -c "$PROBE"; fi
}
pfield() { echo "$1" | awk -v k="$2" '$1==k {$1=""; sub(/^ /,""); print; exit}'; }   # pfield "<probe out>" GPU
route_dev() { ip -o route get "$1" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1; }   # local iface that reaches an IP

# --- memory gate (UMA: a serve relaunched seconds after a teardown gets a PHANTOM "CUDA out of memory" — the
# previous container's GPU pages take ~30-60 s to come back; nothing else is wrong). So after removing old
# containers we WAIT until both boxes report enough available memory, instead of launching into the race.
mem_avail() { free -g | awk '/^Mem:/{print $7}'; }
wait_mem() {  # wait_mem <need GiB> <max seconds>
  local need=$1 max=$2 t=0 h w
  while :; do
    h=$(mem_avail); w=$(ssh_w "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null || echo 0)
    if [ "${h:-0}" -ge "$need" ] && [ "${w:-0}" -ge "$need" ]; then
      echo "  ✓ memory: head ${h}G · worker ${w}G available"; return 0; fi
    if [ "$t" -ge "$max" ]; then
      echo "  ✗ still short after ${max}s: head ${h}G · worker ${w}G available (need ${need}G each)." >&2
      echo "    Another serve on a box? (docker ps on both) — the model needs ~100G per box." >&2; return 1; fi
    [ "$t" = 0 ] && echo "  · waiting for memory to come back (head ${h}G · worker ${w}G, need ${need}G each)…"
    sleep 5; t=$((t+5))
  done
}

# --- firewall probe (no root) ------------------------------------------------------------------------
# Does <from-box> reach <to-box> over the interconnect on an arbitrary high port? A throwaway python listener
# bound to the target's interconnect IP on a random port, then one TCP connect from the other side. Success =
# the target's firewall already admits the peer (ufw rules are per source IP, so one port proves them all) →
# nothing to open, nobody asked for a password. Usage: fw_probe head|worker (the side that must ACCEPT).
fw_probe() {  # fw_probe <listener: head|worker>
  local port=$(( 30000 + RANDOM % 20000 )) lip cmd
  local listener='import socket,sys;s=socket.socket();s.settimeout(8);s.bind((sys.argv[1],int(sys.argv[2])));s.listen(1)
try:
    c,_=s.accept();c.close();print("ok")
except Exception:
    print("none")'
  if [ "$1" = worker ]; then
    lip="$WORKER_IC"
    ssh_w "python3 -c '$listener' $lip $port" > "/tmp/.mbx_fw_$port" 2>/dev/null &
    sleep 1.5
    timeout 4 bash -c ">/dev/tcp/$lip/$port" 2>/dev/null || true
  else
    lip="$HEAD_IC"
    python3 -c "$listener" "$lip" "$port" > "/tmp/.mbx_fw_$port" 2>/dev/null &
    sleep 1.5
    ssh_w "timeout 4 bash -c '>/dev/tcp/$lip/$port'" 2>/dev/null || true
  fi
  wait $! 2>/dev/null
  local r; r="$(cat "/tmp/.mbx_fw_$port" 2>/dev/null)"; rm -f "/tmp/.mbx_fw_$port"
  [ "$r" = ok ]
}

# --- page-cache eviction WITHOUT root -------------------------------------------------------------------
# On a Spark the GPU driver wants pages that are FREE, not merely "available" (reclaimable page cache). After a
# few loads the checkpoint's own shards sit in the page cache (60-70 GB) and MemFree drops to ~1 GB while a new
# load allocates → the driver stalls (a copy that never completes; looks like a hang at 100 % CPU). Root would
# `echo 3 > drop_caches`; we never ask for root. Instead we drop exactly the files we own from the cache with
# POSIX_FADV_DONTNEED via GNU dd — any user may do that to a file they can read. Runs on the head locally and on
# the worker over ssh. Usage: evict_cache <dir>   (all *.safetensors in it)
EVICT='find "$1" -type f -name "*.safetensors" -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null; awk "/^MemFree/{printf \"%d\", \$2/1048576}" /proc/meminfo'
evict_cache() {  # evict_cache <models dir>  (every checkpoint under it) → prints MemFree after eviction
  local h w
  h=$(bash -c "$EVICT" _ "$1")
  w=$(ssh_w "bash -c '$EVICT' _ '$1'" 2>/dev/null || echo "?")
  echo "  · page cache: checkpoint files evicted (no root needed) — MemFree now head ${h}G · worker ${w}G"
}

# --- kernel page compaction (read-only check; the fix needs root → ./tune-host.sh) -----------------------------
# vm.compaction_proactiveness (default 20) lets the kernel migrate pages in the background to build huge blocks.
# On a Spark the GPU's memory IS those pages: on a tightly pinned serve it measured as a 4–5 s slowdown every ~37 s
# (~10 % of throughput). A serving box allocates once at boot and gains nothing from it. Reading needs no privilege.
compaction_check() {
  local h w
  h=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
  w=$(ssh_w "cat /proc/sys/vm/compaction_proactiveness" 2>/dev/null || echo "?")
  if [ "$h" != 0 ] || [ "$w" != 0 ]; then
    echo "  ⚠ vm.compaction_proactiveness is ${h} on the head, ${w} on the worker (want 0): expect ~10 % lower throughput"
    echo "    and periodic 4-5 s stalls under load. One-time fix, needs sudo, shows what it runs first:  ./tune-host.sh"
  else
    echo "  ✓ vm.compaction_proactiveness=0 on both boxes"
  fi
}
