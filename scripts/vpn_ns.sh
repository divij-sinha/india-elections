#!/usr/bin/env bash
# A network namespace whose only route out is its own NordVPN WireGuard (NordLynx) tunnel, so a process run with
# `ip netns exec <name> ...` gets its own VPN IP without touching the rest of the machine.
#   scripts/vpn_ns.sh up <name> [server-hostname]   ## no server: the Nth least-loaded India server, N from the name's digits
#   scripts/vpn_ns.sh down <name>
#   scripts/vpn_ns.sh ip <name>                     ## public IP seen from inside
# Needs root, wireguard-tools (`apt install wireguard-tools`) and the account's NordLynx private key in $NORD_KEY_FILE.
# Get the key once with an access token from the Nord account page (Services > NordVPN > Access token):
#   curl -s -u "token:<TOKEN>" https://api.nordvpn.com/v1/users/services/credentials | jq -r .nordlynx_private_key \
#     > /etc/nordvpn-wg.key && chmod 600 /etc/nordvpn-wg.key
# Don't keep the nordvpn app connected on the same machine: the tunnels' UDP traffic would go through it.
set -euo pipefail

NORD_KEY_FILE="${NORD_KEY_FILE:-/etc/nordvpn-wg.key}"
NORD_COUNTRY_ID="${NORD_COUNTRY_ID:-100}"  ## India
NORD_DNS="103.86.96.100"
CLIENT_ADDR="10.5.0.2/32"  ## NordLynx gives every client this address
API="https://api.nordvpn.com/v1/servers/recommendations?filters\[country_id\]=$NORD_COUNTRY_ID&filters\[servers_technologies\]\[identifier\]=wireguard_udp&limit=30"

cmd="${1:?usage: vpn_ns.sh up|down|ip <name> [server]}"
ns="${2:?namespace name needed}"
wg_if="wg-$ns"

## Prints "hostname station public_key" for the given hostname, or for the Nth least-loaded server
pick_server() {
    curl -s -m 30 "$API" | python3 -c '
import sys, json
want, idx = sys.argv[1], int(sys.argv[2])
servers = []
for s in json.load(sys.stdin):
    keys = [m["value"] for t in s["technologies"] if t["identifier"] == "wireguard_udp" for m in t["metadata"] if m["name"] == "public_key"]
    if keys:
        servers.append((s["hostname"], s["station"], keys[0]))
if want:
    servers = [s for s in servers if s[0] in (want, want + ".nordvpn.com")]
if not servers:
    sys.exit("no matching wireguard server")
print(*servers[idx % len(servers)])
' "$1" "$2"
}

case "$cmd" in
up)
    [ -s "$NORD_KEY_FILE" ] || { echo "missing NordLynx private key in $NORD_KEY_FILE (see top of script)" >&2; exit 1; }
    "$0" down "$ns" 2>/dev/null || true
    idx=$(tr -dc '0-9' <<< "$ns"); idx=$(( ${idx:-1} - 1 ))
    read -r host station pubkey < <(pick_server "${3:-}" "$idx")
    ip netns add "$ns"
    ip -n "$ns" link set lo up
    ## Created here so its UDP socket uses this machine's normal route, then moved inside as the namespace's only way out
    ip link add "$wg_if" type wireguard
    wg set "$wg_if" private-key "$NORD_KEY_FILE" peer "$pubkey" endpoint "$station:51820" allowed-ips 0.0.0.0/0 \
        persistent-keepalive 25
    ip link set "$wg_if" netns "$ns"
    ip -n "$ns" addr add "$CLIENT_ADDR" dev "$wg_if"
    ip -n "$ns" link set "$wg_if" up
    ip -n "$ns" route add default dev "$wg_if"
    mkdir -p "/etc/netns/$ns"
    echo "nameserver $NORD_DNS" > "/etc/netns/$ns/resolv.conf"  ## ip netns exec bind-mounts this over /etc/resolv.conf
    echo "$ns up via $host ($station), public ip $("$0" ip "$ns")"
    ;;
down)
    ip netns del "$ns" 2>/dev/null || true  ## also removes the wireguard interface inside it
    rm -rf "/etc/netns/$ns"
    ;;
ip)
    ip netns exec "$ns" curl -s -m 15 https://api.ipify.org || echo "unreachable"
    ;;
*)
    echo "usage: vpn_ns.sh up|down|ip <name> [server]" >&2; exit 1 ;;
esac
