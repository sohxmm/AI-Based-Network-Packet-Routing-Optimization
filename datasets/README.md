# Datasets

**This project uses synthetic data only.** No external dataset was downloaded,
incorporated, or used to produce any published result.

The previous version of this file listed Mininet, NS-3, Kaggle and CAIDA as
"planned sources". None of them were used. Aspirational scaffolding that reads
like a claim is worse than an empty directory, because a reviewer may reasonably
believe real traces were involved.

## Where the data actually comes from

`core/simulator.py` generates everything:

- a small-world topology (ring + shortcuts + random long-range links),
- link utilization as an AR(1) process around a per-link diurnal cycle,
- congestion bursts, link failures, and load injected by the routing decisions
  themselves.

Regenerate the training data by re-running training; each script builds its own
samples from independently seeded simulators:

```bash
python -m ml.training.train_gnn       # ranking samples
python -m ml.training.train_lstm      # a utilization time series
python -m ml.training.train_rl        # on-policy, generated during training
python -m ml.training.train_regional  # same, per region
```

## Using data that is not synthetic

The platform does not require the simulator. `core/sources.py` defines a
`NetworkSource` interface with three implementations, and everything above it —
the routing algorithms, the QoS engine, the dashboard — consumes a
`NetworkState` without knowing where it came from.

### Replaying a recorded trace

JSONL, one object per timestep:

```json
{"step": 0, "links": [{"source": "A", "target": "B", "base_latency": 12.0,
                       "bandwidth": 1000, "utilization": 0.31}]}
```

or CSV, one row per link per step:

```csv
step,source,target,base_latency,bandwidth,utilization,packet_loss_rate
0,A,B,12.0,1000,0.31,0.0
```

`queue_size` and `packet_loss_rate` are derived from `utilization` when absent,
so a trace only has to carry latency and utilization to be usable. Point the
platform at one:

```bash
curl -X POST localhost:8000/sim/source \
  -H 'Content-Type: application/json' \
  -d '{"kind": "trace", "trace_path": "datasets/example_trace.jsonl"}'
```

A CAIDA, Mininet or NS-3 export converted into either shape will work without
any code change. **That conversion has not been done here**, and no such trace is
included.

### Measuring a real network

`LiveProbeSource` measures the machine the platform runs on, using ordinary
unprivileged ICMP echo via the system `ping` binary. See the *Running against
your own network* section of the root README for how to enable it and what it
can and cannot tell you.

The honest summary: it produces a star topology with exactly one path per
destination, so it demonstrates real telemetry and congestion detection but
**cannot** benchmark routing quality. No number in this project comes from it.
