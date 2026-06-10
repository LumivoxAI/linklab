# Operations And Security

## Trust Model

Linklab v1 assumes a trusted, isolated LAN. It has no authentication, pairing,
authorization, or client identity.

- Do not forward a Linklab port to the Internet.
- Do not send secrets over unauthenticated `ws://`.
- Restrict server ingress with a host firewall and network segmentation.
- Keep `max_connections` and upstream model rate/admission limits finite.
- Log connection origins, admission decisions, and protocol errors without
  logging payloads or unsafe peer-provided diagnostics.

The default server host is `127.0.0.1`; non-loopback or wildcard exposure must be
explicit. `wss://` uses caller-owned `ssl.SSLContext` objects. A trusted or pinned
server certificate can authenticate the server, but Linklab still does not
authenticate clients. Add an authenticated gateway or another protocol version
if client identity is required.

DNS-SD/mDNS announcements are unauthenticated and spoofable. Discovery is only a
routing hint; the WebSocket subprotocol, handshake, and WSS certificate checks
remain authoritative. Multicast may be unavailable across VLANs, containers,
VPNs, and some Wi-Fi networks. Prefer a direct URI when deterministic routing is
required.

## Bounded Resources

All transport and dispatch queues are finite. Important defaults are:

| Resource | Default |
| --- | ---: |
| Client input queue | 16,000 input frames |
| Client playback queue | 2,000 ms |
| Server input-handler queue | 32,000 input frames |
| Server unsent output | 500 ms |
| WebSocket inbound queue | 16 messages |
| WebSocket write limit | 65,536 bytes |
| Server connections | 8 |

Pressure is never hidden by unbounded waiting. Depending on scope it rejects an
activation, aborts input, interrupts playback, cancels a response, ends a
conversation, or closes an unrecoverable connection. `QueueOverflow` from
`OutputWriter.send_audio` means the writer already initiated the protocol's
output-overflow terminal sequence; do not retry that PCM.

Absolute decoder limits cannot be relaxed by peer input: 262,144-byte WebSocket
message, 16,384-byte first hello, nesting depth 8, 64 map entries, 32 array
items, 65,536 UTF-8 bytes per string, 64 bytes per identifier/reason/type, and
16,384 bytes for non-audio binary values. Compression is disabled.

## Timeouts, Keepalive, And Reconnect

Transport defaults are connect 10 s, handshake 5 s, ping interval 20 s, pong
timeout 20 s, and close 10 s. Production keepalive cannot be disabled. Server
defaults are waiting 60 s, open input 120 s, processing 120 s, and negotiated
idle 120 s.

Optional client reconnect uses full-jitter exponential backoff from 0.5 s to
30 s. It follows normal close 1001 or abnormal transport loss, but not protocol
1002 or policy 1008. Linklab never resumes IDs, conversations, or PCM and never
replays audio. Activation while disconnected is rejected.

## Shutdown

Use `close()` or async context managers. Client shutdown sends required
conversation cancellation and playback accounting before close when possible.
Server shutdown stops admission/discovery, terminates sessions, and uses close
code 1001. Shutdown is bounded and idempotent. Do not stop the event loop before
`close()` completes.

## Observability

Pass a `lumivox_core.logger.Logger` to each facade. Linklab calls `bind`, `debug`,
`info`, `warning`, `error`, and `exception` directly and does not configure
process logging or suppress logger failures.

Structured records cover listener/discovery lifecycle, admission, handshake,
connection/session state, reconnect, protocol failures, callback/handler
failures, queue occupancy/overflow/residence, IDs, frame ranges, cancellation
reasons, stage latency, and Ping/Pong RTT. Normal records never contain PCM or
full transcript/response text. Keep per-audio and per-token diagnostics at a
rate-limited debug level in the surrounding application as well.

## Failure Boundaries

Malformed data, limits, unknown messages, ID mismatches, and impossible lifecycle
transitions are connection-fatal. Model and application failures use the
narrowest input, response, or conversation scope. Client callback failures and
server handler failures are converted to defined terminal outcomes rather than
silently killing dispatch tasks. Treat connection close 1002 as a compatibility
or implementation fault that must be investigated, not retried indefinitely.
