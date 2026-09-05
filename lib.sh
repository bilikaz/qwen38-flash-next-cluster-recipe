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

# --- memory sanity (UMA: a serve relaunched seconds after a teardown sees phantom OOMs) ---------------
warn_mem() {  # warn_mem <label> <free GiB>
  if [ -n "$2" ] && [ "$2" -lt 100 ]; then
    echo "  ⚠ $1: only ${2}G free — the model needs ~100G per box. Another serve running? Just stopped one?" >&2
    echo "    (unified memory takes ~30 s to come back after a container stops — wait, then retry)" >&2
  fi
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
