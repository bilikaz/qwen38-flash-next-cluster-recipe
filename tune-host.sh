#!/usr/bin/env bash
# tune-host.sh — the ONE host setting this kit recommends but never applies on its own (it needs root):
#
#   vm.compaction_proactiveness = 0      on both boxes, persisted in /etc/sysctl.d/99-myllmbox-compaction.conf
#
# WHY: the kernel's background page compactor migrates pages to build large contiguous blocks. On a DGX Spark the
# GPU's memory IS ordinary system pages, so every migrated page is first unmapped from the GPU. On a serve pinned
# close to the memory edge (this one) that measured as a 4–5 s slowdown every ~37 s — about 10 % of throughput and a
# 30 % drop in the worst 10-second window. A serving box allocates once at boot; it gains nothing from the upkeep.
# Direct compaction on a real allocation failure still works — only the proactive background pass is disabled.
#
# This script shows exactly what it will run, asks once, then lets sudo prompt the normal way on each box (the
# password is typed into sudo's own prompt — never read, stored or passed by this script). Reversible:
#   sudo sysctl -w vm.compaction_proactiveness=20 && sudo rm /etc/sysctl.d/99-myllmbox-compaction.conf
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source lib.sh
load_cluster || { echo "✗ no cluster.env yet — run ./setup.sh (or ./run.sh) first"; exit 1; }

CMD='printf "vm.compaction_proactiveness = 0\n" > /etc/sysctl.d/99-myllmbox-compaction.conf && sysctl -q vm.compaction_proactiveness=0 && echo "  ✓ $(hostname): vm.compaction_proactiveness=$(cat /proc/sys/vm/compaction_proactiveness) (persisted)"'

h=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
w=$(ssh_w "cat /proc/sys/vm/compaction_proactiveness" 2>/dev/null || echo "?")
echo "current: head=${h}  worker(${WORKER_HOST})=${w}   (want 0 on both)"
if [ "$h" = 0 ] && [ "$w" = 0 ]; then echo "✓ already set on both boxes — nothing to do"; exit 0; fi
echo
echo "This will run AS ROOT on each box that is not yet 0:"
echo "  sudo bash -c '$CMD'"
echo
read -rp "Proceed? [y/N] " ans; [[ "${ans:-N}" =~ ^[Yy]$ ]] || { echo "aborted — nothing changed"; exit 0; }
[ "$h" = 0 ] || { echo "── head ──";   sudo bash -c "$CMD"; }
[ "$w" = 0 ] || { echo "── worker ──"; ssh_wt "sudo bash -c '$CMD'"; }
echo "done. It is live now (no restart) and survives reboots."
