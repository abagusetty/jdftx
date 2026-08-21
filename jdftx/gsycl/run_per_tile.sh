#!/bin/bash
# run_per_tile.sh -- launch independent JDFTx GPU runs, one per Intel PVC tile.
#
# Production model (deliberately MPI-free): JDFTx runs as N completely separate
# single-rank processes, one bound to each GPU tile. A tile is its own 64 GB
# HBM domain and each input is a different physical system, so there is nothing
# for MPI to coordinate; leaving it out removes the whole rank-placement and
# oversubscription problem class. Use this to run a *batch* of independent
# inputs concurrently, not to make a single input faster.
#
# What gets rationed, and why:
#   GPU  -- one tile per process via ZE_AFFINITY_MASK, which filters Level Zero
#           enumeration so the process sees exactly one device.
#   CPU  -- the physical cores this job actually owns (the cpuset, which on this
#           node excludes core 0 of each socket) are split evenly among the
#           tiles served by the same socket, and each process is bound to its
#           own disjoint core range on the socket its GPU is attached to (both
#           the GPU-to-socket map and the cpuset are read, not assumed).
#           Threads are capped to that count: JDFTx sizes its own
#           std::thread pool from OMP_NUM_THREADS, and inheriting this node's
#           OMP_NUM_THREADS=208 makes every process spawn 208 threads per
#           operator -- 12 of those is what exhausts the node's process table.
#   DRAM -- memory is bound to the local NUMA node (no cross-socket traffic and
#           no single process eating the whole node's memory), plus a per-process
#           RLIMIT_DATA cap at an equal share of usable DRAM. The cap turns a
#           runaway job into one clean std::bad_alloc in that job instead of the
#           OOM killer picking off its 11 healthy neighbours.
#
# Usage:
#   ./run_per_tile.sh <jdftx_gpu binary> <outdir> <input1.in> [input2.in ...]
#
# Inputs are dispatched to whichever tile is free; with more inputs than tiles,
# the rest queue.
#
# Env overrides:
#   NTILES          number of tiles to use         (default: autodetected)
#   CORES_PER_TILE  CPU threads per process        (default: physical cores/tiles)
#   MEM_PER_TILE_GB per-process DRAM cap in GiB    (default: usable DRAM/tiles)
#                   set to 0 to disable the cap
#   MEM_FRACTION    fraction of total DRAM treated as usable (default: 0.85)
#   PSEUDO          JDFTX_PSEUDO path              (default: <bindir>/pseudopotentials)
#   HIERARCHY       COMPOSITE | FLAT               (default: COMPOSITE)
#   JDFTX_ARGS      extra args appended to each jdftx invocation
#   DRY_RUN=1       print the plan and exit
#   DEBUG_SERIALIZE=1  add ZE_SERIALIZE=2 (DEBUG ONLY -- see note)
#
# NOTE ON ZE_SERIALIZE: ZE_SERIALIZE=2 makes the Level Zero driver block on every
# command-list submission. It is a debugging aid for deciding whether a numerical
# discrepancy is a race (if results change when it is set, you have one). It
# serializes all GPU work and costs a large amount of performance, so it is off
# unless DEBUG_SERIALIZE=1 is passed explicitly. Never set it in production.

set -u -o pipefail

usage() { sed -n '2,/^$/p' "$0"; exit 1; }
[ $# -lt 3 ] && usage

BIN="$1"; shift
OUTDIR="$1"; shift
INPUTS=("$@")

[ -x "$BIN" ] || { echo "error: '$BIN' is not executable" >&2; exit 1; }
BIN="$(readlink -f "$BIN")"
mkdir -p "$OUTDIR" || exit 1
OUTDIR="$(readlink -f "$OUTDIR")"

BINDIR="$(dirname "$BIN")"
PSEUDO="${PSEUDO:-$BINDIR/pseudopotentials}"
HIERARCHY="${HIERARCHY:-COMPOSITE}"
JDFTX_ARGS="${JDFTX_ARGS:-}"

#--- GPU topology ---------------------------------------------------------
# One entry per GPU, giving the NUMA node it hangs off. Read from sysfs rather
# than assumed: the socket split is a property of the PCIe layout, and a wrong
# guess silently costs every memory access a cross-socket hop.
GPU_NUMA=()
for card in /sys/class/drm/card[0-9]*; do
	[ -e "$card/device/numa_node" ] || continue
	n=$(cat "$card/device/numa_node")
	[ "$n" -lt 0 ] 2>/dev/null && n=0   #-1 means "not reported"
	GPU_NUMA+=("$n")
done
NGPUS=${#GPU_NUMA[@]}
if [ "$NGPUS" -eq 0 ]; then    #no sysfs view (e.g. inside a container): assume an Aurora node
	NGPUS=6; GPU_NUMA=(0 0 0 1 1 1)
fi
TILES_PER_GPU=2                #every PVC Max 1550 is two tiles
MAXTILES=$(( NGPUS * TILES_PER_GPU ))
NTILES="${NTILES:-$MAXTILES}"
# Clamp an explicit override: past MAXTILES the tile index would name a device
# that does not exist, and ZE_AFFINITY_MASK would silently fall back to tile 0,
# quietly oversubscribing one tile with every surplus process.
if [ "$NTILES" -gt "$MAXTILES" ]; then
	echo "warning: NTILES=$NTILES exceeds the ${MAXTILES} tiles on this node; using ${MAXTILES}" >&2
	NTILES=$MAXTILES
fi
[ "$NTILES" -lt 1 ] && NTILES=1

#--- CPU topology ---------------------------------------------------------
# Physical cores only, listed per NUMA node. JDFTx gains nothing from SMT and
# the sibling threads are better left free than handed out as if they were cores.
#
# Only CPUs inside this job's cpuset may be used. That is not the whole node:
# core specialization here reserves core 0 of each socket for the OS, so CPUs
# 0 and 52 (and their SMT siblings 104/156) are outside it. numactl refuses to
# start a process bound to a CPU it does not own -- "<0,1,2,...> is invalid" --
# so a pool built from lscpu alone silently kills the first tile on each socket.
declare -A ALLOWED
while read -r c; do ALLOWED[$c]=1; done < <(
	awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status 2>/dev/null | tr ',' '\n' |
	while read -r spec; do
		case "$spec" in
			"") ;;
			*-*) seq "${spec%%-*}" "${spec##*-}";;
			*) echo "$spec";;
		esac
	done)

declare -A NODE_CORES              #numa node -> space separated physical cpu ids
declare -A NODE_NTILES             #numa node -> number of tiles served
seen_core=""
while IFS=, read -r cpu core node; do
	case "$cpu" in \#*|"") continue;; esac
	# One logical CPU per physical core, and only one we are allowed to use:
	# skip a core's first sibling if it is reserved and take the next instead.
	[ ${#ALLOWED[@]} -gt 0 ] && [ -z "${ALLOWED[$cpu]:-}" ] && continue
	case " $seen_core " in *" $node:$core "*) continue;; esac
	seen_core="$seen_core $node:$core"
	NODE_CORES[$node]="${NODE_CORES[$node]:-} $cpu"
done < <(lscpu -p=CPU,CORE,NODE 2>/dev/null)

for (( t=0; t<NTILES; t++ )); do
	g=$(( t / TILES_PER_GPU ))
	node=${GPU_NUMA[$(( g % NGPUS ))]}
	NODE_NTILES[$node]=$(( ${NODE_NTILES[$node]:-0} + 1 ))
done

#--- DRAM budget ----------------------------------------------------------
MEM_FRACTION="${MEM_FRACTION:-0.85}"
TOTAL_MEM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
if [ -n "${MEM_PER_TILE_GB:-}" ]; then
	MEM_CAP_KB=$(( MEM_PER_TILE_GB * 1024 * 1024 ))
else
	MEM_CAP_KB=$(awk -v t="$TOTAL_MEM_KB" -v f="$MEM_FRACTION" -v n="$NTILES" \
		'BEGIN { printf "%d", (t*f)/n }')
	MEM_PER_TILE_GB=$(( MEM_CAP_KB / 1024 / 1024 ))
fi

#--- Per-tile core assignment --------------------------------------------
# Each tile gets a disjoint slice of its own socket's physical cores. The
# remainder is spread over the first tiles on that socket rather than dropped,
# so no core sits idle while another process is oversubscribed.
declare -A TILE_CPUS TILE_NODE TILE_MASK
declare -A NODE_TAKEN
for (( t=0; t<NTILES; t++ )); do
	g=$(( t / TILES_PER_GPU )); s=$(( t % TILES_PER_GPU ))
	node=${GPU_NUMA[$(( g % NGPUS ))]}
	read -r -a cores <<< "${NODE_CORES[$node]:-}"
	nTilesHere=${NODE_NTILES[$node]}
	nCoresHere=${#cores[@]}
	if [ "$nCoresHere" -eq 0 ]; then cores=($t); nCoresHere=1; fi
	idx=${NODE_TAKEN[$node]:-0}                       #which tile-on-this-node this is
	base=$(( nCoresHere / nTilesHere ))
	extra=$(( nCoresHere % nTilesHere ))
	take=$(( base + (idx < extra ? 1 : 0) ))
	[ "$take" -lt 1 ] && take=1
	# start offset = sum of the takes of the earlier tiles on this node
	start=$(( idx * base + (idx < extra ? idx : extra) ))
	slice=("${cores[@]:$start:$take}")
	[ ${#slice[@]} -eq 0 ] && slice=("${cores[0]}")
	TILE_CPUS[$t]=$(IFS=,; echo "${slice[*]}")
	TILE_NODE[$t]=$node
	if [ "$HIERARCHY" = "FLAT" ]; then TILE_MASK[$t]="$t"; else TILE_MASK[$t]="$g.$s"; fi
	NODE_TAKEN[$node]=$(( idx + 1 ))
done

DEFAULT_CPT=$(awk -F, '{print NF}' <<< "${TILE_CPUS[0]}")
CORES_PER_TILE="${CORES_PER_TILE:-0}"   #0 = per-tile value from the table above

cat <<EOF
plan: ${NTILES} tiles on ${NGPUS} GPUs, ${#GPU_NUMA[@]}-card sysfs topology, hierarchy ${HIERARCHY}
      $(( TOTAL_MEM_KB / 1024 / 1024 )) GiB DRAM total -> ${MEM_PER_TILE_GB} GiB/process cap$( [ "$MEM_CAP_KB" -eq 0 ] && echo " (disabled)" )
      $( [ "$CORES_PER_TILE" -eq 0 ] && echo "up to ${DEFAULT_CPT}" || echo "${CORES_PER_TILE}" ) threads/process (exact split per tile below)
      pseudopotentials: ${PSEUDO}
EOF
for (( t=0; t<NTILES; t++ )); do
	printf '      tile %-5s numa %s  cpus %s\n' "${TILE_MASK[$t]}" "${TILE_NODE[$t]}" "${TILE_CPUS[$t]}"
done
[ "${DEBUG_SERIALIZE:-0}" = "1" ] && echo "WARNING: ZE_SERIALIZE=2 set -- DEBUG ONLY, not performance-representative"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

#--- Launch ---------------------------------------------------------------
run_one()
{	local tile=$1 input=$2
	local base; base="$(basename "${input%.in}")"
	local mask="${TILE_MASK[$tile]}" cpus="${TILE_CPUS[$tile]}" node="${TILE_NODE[$tile]}"
	local nthreads=$CORES_PER_TILE
	[ "$nthreads" -eq 0 ] && nthreads=$(awk -F, '{print NF}' <<< "$cpus")
	local rundir="$OUTDIR/$base"
	mkdir -p "$rundir"

	(
		# RLIMIT_DATA rather than RLIMIT_AS: it covers the heap and private
		# anonymous mappings (where JDFTx's host-side arrays live) without
		# counting the large file-backed VA ranges the Level Zero driver maps
		# for device USM, which an RLIMIT_AS cap would trip on immediately.
		[ "$MEM_CAP_KB" -gt 0 ] && ulimit -d "$MEM_CAP_KB" 2>/dev/null
		export ZE_AFFINITY_MASK="$mask"
		export ZE_FLAT_DEVICE_HIERARCHY="$HIERARCHY"
		export OMP_NUM_THREADS="$nthreads"
		export MKL_NUM_THREADS="$nthreads"
		export JDFTX_PSEUDO="$PSEUDO"
		[ "${DEBUG_SERIALIZE:-0}" = "1" ] && export ZE_SERIALIZE=2
		cd "$rundir" || exit 1
		echo "[tile $mask | numa $node | cpus $cpus | ${nthreads}t] $base"
		local -a pin=()
		if command -v numactl >/dev/null 2>&1; then
			pin=(numactl --physcpubind="$cpus" --membind="$node")
		elif command -v taskset >/dev/null 2>&1; then
			pin=(taskset -c "$cpus")
		fi
		# -c pins JDFTx's own thread count explicitly; it overrides
		# OMP_NUM_THREADS, so the two are set to the same value on purpose.
		exec "${pin[@]}" "$BIN" -c "$nthreads" -i "$input" -o "$base.out" $JDFTX_ARGS
	) > "$rundir/launch.log" 2>&1 &
	# Return the pid through a global, NOT stdout: $(run_one ...) would run this
	# in a command-substitution subshell, making the job a grandchild that the
	# main shell can no longer wait on.
	LAST_PID=$!
}

declare -A SLOT_PID SLOT_NAME
# ACTIVE mirrors ${#SLOT_PID[@]}: under `set -u`, bash 4.4 treats *that* length
# expansion on a declared-but-empty associative array as an unbound variable and
# aborts. (The key expansion "${!SLOT_PID[@]}" is fine when empty, and must be
# written plainly -- the ${!arr[@]+...} guard form parses as indirection and
# fails with "bad substitution" as soon as the array is non-empty.)
ACTIVE=0
fail=0; done_count=0; skipped=0

reap_one()   #block until some tile finishes, then free its slot
{	local t pid running
	while true; do
		# `jobs -rp` lists only still-running jobs, so a finished-but-unreaped
		# child drops off it. kill -0 cannot be used here: a zombie child still
		# answers it. `wait -n` cannot either -- this bash (4.4) has no -p, so it
		# reaps a job without telling us which, and the later per-pid wait then
		# fails with "not a child" and every run looks like a failure.
		running=" $(jobs -rp | tr '\n' ' ') "
		for t in "${!SLOT_PID[@]}"; do
			pid=${SLOT_PID[$t]}
			case "$running" in *" $pid "*) continue;; esac
			if wait "$pid"; then
				echo "  done: ${SLOT_NAME[$t]} (tile ${TILE_MASK[$t]})"
			else
				echo "  FAILED: ${SLOT_NAME[$t]} (tile ${TILE_MASK[$t]}) -- see $OUTDIR/${SLOT_NAME[$t]}/launch.log" >&2
				fail=$(( fail + 1 ))
			fi
			done_count=$(( done_count + 1 ))
			unset "SLOT_PID[$t]" "SLOT_NAME[$t]"
			ACTIVE=$(( ACTIVE - 1 ))
			return 0
		done
		[ "$ACTIVE" -eq 0 ] && return 1
		sleep 1
	done
}

for input in "${INPUTS[@]}"; do
	if [ ! -f "$input" ]; then
		echo "skip: no such input '$input'" >&2
		skipped=$(( skipped + 1 ))
		continue
	fi
	input="$(readlink -f "$input")"
	# find a free tile, waiting for one if all are busy
	free=-1
	while [ "$free" -lt 0 ]; do
		for (( t=0; t<NTILES; t++ )); do
			[ -z "${SLOT_PID[$t]:-}" ] && { free=$t; break; }
		done
		[ "$free" -lt 0 ] && reap_one
	done
	run_one "$free" "$input"
	SLOT_PID[$free]=$LAST_PID
	SLOT_NAME[$free]="$(basename "${input%.in}")"
	ACTIVE=$(( ACTIVE + 1 ))
done

while [ "$ACTIVE" -gt 0 ]; do reap_one || break; done

echo
if [ "$skipped" -gt 0 ]; then echo "${skipped} input(s) skipped (not found)" >&2; fi
if [ "$done_count" -eq 0 ]; then
	echo "nothing ran" >&2
	exit 1
fi
if [ "$fail" -eq 0 ]; then
	echo "all ${done_count} run(s) completed successfully"
else
	echo "${fail} of ${done_count} run(s) FAILED" >&2
fi
exit $(( fail > 0 ? 1 : 0 ))
