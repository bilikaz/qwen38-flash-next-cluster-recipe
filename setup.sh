#!/usr/bin/env bash
# setup.sh — wire two DGX Sparks into one serving cluster. Run it ON THE HEAD (the box that will serve the API).
#
# It asks for the second box (ssh IP + user, password once), installs an ssh key, probes BOTH boxes (network
# interfaces, RDMA devices, GPU, docker), DISCOVERS the interconnect link between them (the ConnectX cable/switch:
# the interface pair that actually reaches the other box, not the management LAN), opens the firewall for the
# peer's interconnect IP on both boxes, creates the models/cache dirs on the worker, and writes cluster.env.
# ./run.sh calls this automatically when cluster.env is missing. Idempotent — rerun after re-cabling.
#
#   ./setup.sh                 # interactive
#   ./setup.sh user@10.0.0.2   # worker given on the command line
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source lib.sh

echo "── myllmbox 2-Spark cluster setup ──"
echo "This machine is the HEAD (runs the API). I need the SECOND Spark (the worker)."
if [ -n "${1:-}" ]; then
  W="$1"
else
  read -rp "  worker ssh IP: " whost
  [ -n "$whost" ] || { echo "no worker given — aborting"; exit 1; }
  read -rp "  worker ssh user [$USER]: " wuser; wuser="${wuser:-$USER}"
  W="$wuser@$whost"
fi
WORKER_USER="${W%@*}"; WORKER_HOST="${W#*@}"
[ "$WORKER_USER" != "$W" ] || WORKER_USER="$USER"

# 1. reach the worker at all (port 22) BEFORE touching keys — a typo'd IP must fail here, loudly, not inside
#    ssh-copy-id's retries.
if ! timeout 5 bash -c ">/dev/tcp/$WORKER_HOST/22" 2>/dev/null; then
  echo "✗ nothing answers on $WORKER_HOST:22 — wrong IP (typo?), box off, or ssh blocked by its firewall"
  echo "  (on the worker: sudo ufw allow OpenSSH). Rerun ./setup.sh with the right address."
  exit 1
fi
# 2. passwordless ssh already works? then we install NOTHING. Otherwise install exactly ONE key — the kit's own
#    ed25519 — never `ssh-copy-id` with no -i (it would push every key it finds in ~/.ssh).
if ssh -o BatchMode=yes -o ConnectTimeout=8 "$W" true 2>/dev/null; then
  echo "✓ ssh $W (key already works — nothing installed)"
else
  KEY=~/.ssh/id_ed25519
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -f "$KEY" -C "myllmbox-cluster-$(hostname)"
  echo "· installing $KEY.pub on $W (its password, once):"
  ssh-copy-id -i "$KEY.pub" -o ConnectTimeout=8 "$W" >/dev/null
  ssh -o BatchMode=yes -o ConnectTimeout=8 -i "$KEY" "$W" true || { echo "✗ still cannot ssh to $W"; exit 1; }
  echo "✓ ssh $W"
fi

# 3. probe both boxes
echo "· probing head (this machine)…";  PH="$(probe)"
echo "· probing worker $W…";            PW="$(probe "$W")"
for side in HEAD WORKER; do
  P="$([ $side = HEAD ] && echo "$PH" || echo "$PW")"
  echo "  [$side] GPU: $(pfield "$P" GPU) · $(pfield "$P" DOCKER) · nvidia runtime: $(pfield "$P" NVRT) · /dev/infiniband: $(pfield "$P" RDMA) · free $(pfield "$P" MEM)G"
  [ "$(pfield "$P" DOCKER)" != none ] || { echo "✗ $side has no docker"; exit 1; }
  [ "$(pfield "$P" NVRT)" = yes ]     || echo "  ⚠ $side: docker does not list the nvidia runtime — install nvidia-container-toolkit"
  [ "$(pfield "$P" RDMA)" = yes ]     || echo "  ⚠ $side: no /dev/infiniband — NCCL will fall back to TCP (~2x slower steps). Install rdma-core + check the ConnectX link."
done

# 4. interconnect discovery. Management = the path ssh took (head iface that routes to WORKER_HOST; on the
#    worker, the iface holding WORKER_HOST). Candidates = every OTHER interface. Try every head-candidate →
#    worker-candidate pair with a bound ping; the pairs that answer are the interconnect links.
echo "· discovering the interconnect…"
HEAD_MGMT_IF="$(route_dev "$WORKER_HOST")"
mapfile -t HC < <(echo "$PH" | awk -v m="$HEAD_MGMT_IF" '$1=="IFACE" && $2!=m {print $2, $3, $4}')
mapfile -t WC < <(echo "$PW" | awk -v h="$WORKER_HOST" '$1=="IFACE" && $3!=h {print $2, $3, $4}')
LINKS=()
for h in "${HC[@]}"; do
  set -- $h; hif=$1; hip=$2; hhca=$3
  for w in "${WC[@]}"; do
    set -- $w; wif=$1; wip=$2; whca=$3
    if ping -I "$hif" -c1 -W1 "$wip" >/dev/null 2>&1; then
      LINKS+=("$hif $hip $hhca $wif $wip $whca")
      echo "  link: head $hif ($hip$([ "$hhca" != - ] && echo ", RDMA $hhca")) ⇄ worker $wif ($wip$([ "$whca" != - ] && echo ", RDMA $whca"))"
    fi
  done
done
if [ "${#LINKS[@]}" -eq 0 ]; then
  echo "  ⚠ no dedicated interconnect found — falling back to the management LAN (no RDMA; make sure it does not block ports)."
  HIP="$(echo "$PH" | awk -v m="$HEAD_MGMT_IF" '$1=="IFACE" && $2==m {print $3; exit}')"
  WIF="$(echo "$PW" | awk -v h="$WORKER_HOST" '$1=="IFACE" && $3==h {print $2; exit}')"
  PICK="$HEAD_MGMT_IF $HIP - $WIF $WORKER_HOST -"
elif [ "${#LINKS[@]}" -eq 1 ]; then
  PICK="${LINKS[0]}"
else
  echo "  several links — pick the one to use (a dedicated ConnectX link beats a shared switch):"
  for i in "${!LINKS[@]}"; do echo "    [$((i+1))] ${LINKS[$i]}"; done
  read -rp "  choice [1]: " sel; PICK="${LINKS[$(( ${sel:-1} - 1 ))]}"
fi
set -- $PICK
HEAD_IFACE=$1; HEAD_IC=$2; HEAD_HCA=$3; WORKER_IFACE=$4; WORKER_IC=$5; WORKER_HCA=$6
[ "$HEAD_HCA" = - ] && HEAD_HCA=""; [ "$WORKER_HCA" = - ] && WORKER_HCA=""
echo "✓ interconnect: head $HEAD_IFACE $HEAD_IC${HEAD_HCA:+ (RDMA $HEAD_HCA)} ⇄ worker $WORKER_IFACE $WORKER_IC${WORKER_HCA:+ (RDMA $WORKER_HCA)}"
[ -n "$HEAD_HCA" ] && [ -n "$WORKER_HCA" ] || echo "  ⚠ no RDMA device on this link — NCCL runs over TCP sockets (works, ~2x slower per step)"
# 4b. the second PCIe half of the card. Both boxes need it ACTIVE with an IPv4 before NCCL may use it; then the pair
#     goes into cluster.env as a comma list and run.sh stripes NCCL over both (~13 → ~20 GB/s). Otherwise: say exactly
#     what to do (root, once per box — this script never runs it) and pin the one device that works.
WORKER="$W"
if [ -n "$HEAD_HCA" ] && [ -n "$WORKER_HCA" ]; then
  read -r _ HSIB HSND HSIP < <(hca_siblings head "$HEAD_HCA" | grep '^SIB ' | head -1 || true)
  read -r _ WSIB WSND WSIP < <(hca_siblings worker "$WORKER_HCA" | grep '^SIB ' | head -1 || true)
  if [ -n "${HSIB:-}" ] && [ -n "${WSIB:-}" ]; then
    if [ "${HSIP:-"-"}" != "-" ] && [ "${WSIP:-"-"}" != "-" ]; then
      HEAD_HCA="$HEAD_HCA,$HSIB"; WORKER_HCA="$WORKER_HCA,$WSIB"
      echo "✓ both PCIe halves of the card have an address — head $HEAD_HCA ⇄ worker $WORKER_HCA (NCCL stripes over both)"
    else
      echo "  ⚠ each card has a second PCIe half (head $HSIB on $HSND, worker $WSIB on $WSND) — ~13 GB/s more, unused until its"
      echo "    interface has an IPv4 on BOTH boxes. Root, once, on each box that shows '-' below, then rerun ./setup.sh:"
      echo "      head   $HSND ${HSIP:-"-"}   worker $WSND ${WSIP:-"-"}"
      echo "      sudo nmcli con mod \"\$(nmcli -t -f NAME,DEVICE con show | grep ':<iface>\$' | cut -d: -f1)\" ipv4.method link-local ipv6.method disabled"
      echo "      sudo nmcli con up  \"\$(nmcli -t -f NAME,DEVICE con show | grep ':<iface>\$' | cut -d: -f1)\""
    fi
  fi
fi

# 5. firewall — test the EFFECT without root: can each box reach the other over the interconnect on a high port?
#    (a throwaway listener + one connect; ufw rules are per source IP so one port proves them all). Only a box that
#    actually BLOCKS its peer gets a fix, and only after showing the exact command and asking. Nobody types a
#    password for a check.
echo "· firewall: probing whether the boxes already admit each other over the interconnect…"
WORKER="$W"
open_rule() {  # open_rule head|worker <peer-ip>  — consent-gated `ufw allow from <peer>` on that box
  local box="$1" peer="$2" cmd="sudo ufw allow from $peer"
  echo "  ✗ $box blocks its peer ($peer). The fix is ONE firewall rule on $box:"
  echo "        $cmd"
  read -rp "  run it now on $box (asks for $box's sudo password)? [y/N]: " ok
  case "$ok" in
    [Yy]*) if [ "$box" = head ]; then $cmd >/dev/null && echo "  ✓ head: allowed $peer"
           else ssh_wt "$cmd >/dev/null" && echo "  ✓ worker: allowed $peer"; fi ;;
    *) echo "  · skipped — run it yourself on $box before ./run.sh, or the cluster will hang at NCCL init";;
  esac
}
if fw_probe worker; then echo "  ✓ worker admits the head ($HEAD_IC) — nothing to open"; else open_rule worker "$HEAD_IC"; fi
if fw_probe head;   then echo "  ✓ head admits the worker ($WORKER_IC) — nothing to open"; else open_rule head "$WORKER_IC"; fi

# 6. dirs on the worker at the SAME absolute paths as here (weights are synced there; caches are per box)
MODELS_ABS="$(mkdir -p "$(rkey server models_dir)" && cd "$(rkey server models_dir)" && pwd)"
CACHE_ABS="$(mkdir -p "$(rkey server cache_dir)" && cd "$(rkey server cache_dir)" && pwd)"
ssh_w "mkdir -p '$MODELS_ABS' '$CACHE_ABS'"
echo "✓ worker dirs: $MODELS_ABS  $CACHE_ABS"

# 7. write cluster.env
cat > "$CLUSTER_ENV" <<EOF
# written by setup.sh $(date '+%Y-%m-%d %H:%M') — this cluster (machine-specific, gitignored). Rerun ./setup.sh to redo.
HEAD_IC=$HEAD_IC            # head interconnect IP (NCCL/gloo/rendezvous bind here)
HEAD_IFACE=$HEAD_IFACE
HEAD_HCA=$HEAD_HCA          # RDMA device(s) on the head's link, comma list = both PCIe halves ('' = TCP)
WORKER_HOST=$WORKER_HOST    # worker ssh address (control plane)
WORKER_USER=$WORKER_USER
WORKER_IC=$WORKER_IC        # worker interconnect IP
WORKER_IFACE=$WORKER_IFACE
WORKER_HCA=$WORKER_HCA
EOF
echo "✓ wrote $CLUSTER_ENV"
echo
echo "── cluster ready: ./run.sh serves the model across both boxes ──"
