# Architecture

TODO: implement

The planned system architecture is:

1. Network simulator produces dynamic router/link state.
2. Routing algorithms evaluate candidate paths.
3. AI/ML components predict congestion and learn routing policies.
4. FastAPI exposes REST and WebSocket APIs.
5. PostgreSQL stores routing events and network snapshots.
6. React visualizes topology, metrics, congestion, and algorithm comparisons.
