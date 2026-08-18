# Congestion-Aware Packet Routing: What This System Does, Why, and What It Taught Us

*A study guide written to be read end-to-end, in the shape of a research paper.*

---

## Abstract

We built a controlled benchmarking platform for congestion-aware network
routing. It implements eight routing strategies — two exact classical solvers, a
constraint-aware classical baseline, one metaheuristic, three learned policies,
and a random floor — and measures them head-to-head under seven reproducible
network scenarios with quality-of-service constraints, statistical significance
testing across independent replications, and automated detection of the ways
each method can silently fail.

The project's central finding is not a performance number. It is a structural
one: **on a single additive cost, learned routing cannot beat Dijkstra, because
Dijkstra is provably optimal there.** A well-trained graph neural network in our
system reproduces Dijkstra's chosen path essentially 100% of the time on
best-effort traffic, and this is *correct behaviour*, not a bug. Learned methods
can only add value where the objective is not a single additive cost — under
multi-constrained quality-of-service requirements, or in a network that reacts to
the routing decisions made in it. Both conditions were absent from the original
design, and creating them was the largest single change we made.

Along the way we recovered a set of lessons about *evaluation* that we think are
more transferable than anything about routing: a metric without a floor and a
ceiling conveys no information; correlated observations are not independent
samples; a model that cannot beat the trivial baseline should not be shipped; and
instrumentation that reports a problem but does not enforce anything will
eventually be ignored.

---

## Contents

**Part I — Background**
1. [The problem](#1-the-problem)
2. [Routing theory you need](#2-routing-theory-you-need)
3. [Why "AI beats Dijkstra" is harder than it sounds](#3-why-ai-beats-dijkstra-is-harder-than-it-sounds)

**Part II — The system**
4. [Architecture](#4-architecture)
5. [The simulator: designing an environment that can answer the question](#5-the-simulator)
6. [The cost model](#6-the-cost-model)
7. [Quality of service: making the problem genuinely hard](#7-quality-of-service)

**Part III — The algorithms**
8. [Dijkstra](#8-dijkstra)
9. [Bellman-Ford](#9-bellman-ford)
10. [Constrained k-shortest paths](#10-constrained-k-shortest-paths)
11. [Ant Colony Optimization](#11-ant-colony-optimization)
12. [Graph neural network](#12-graph-neural-network)
13. [Reinforcement learning](#13-reinforcement-learning)
14. [Multi-agent reinforcement learning](#14-multi-agent-reinforcement-learning)
15. [LSTM congestion forecasting](#15-lstm-congestion-forecasting)

**Part IV — Method and results**
16. [Experimental methodology](#16-experimental-methodology)
17. [Results](#17-results)
18. [Discussion](#18-discussion)

**Part V — The world**
19. [Real-world applications](#19-real-world-applications)
20. [How you would actually deploy this](#20-how-you-would-actually-deploy-this)
21. [Limitations and future work](#21-limitations-and-future-work)
22. [Glossary](#22-glossary)
23. [Further reading](#23-further-reading)

---

# Part I — Background

## 1. The problem

A network is a graph. Routers are nodes, links are edges, and a packet travelling
from A to B must be assigned a path through that graph. The question "which
path?" sounds simple and is not, because the *cost* of a link is not fixed.

A link's latency depends on how loaded it is. An idle 10 ms link is a 10 ms link;
the same link at 95% utilization has a queue in front of it and may cost several
times that. Load, in turn, depends on the routing decisions being made — which is
the feedback loop that makes the problem interesting.

Traditional protocols (RIP, OSPF, static routing) largely ignore this. They
compute shortest paths over *static* or slowly-updated weights, converge, and
then keep using the result until something changes structurally. Under bursty,
shifting load they:

- react slowly, because reconvergence is expensive and deliberately damped;
- create hotspots, because every router independently agrees the same link is
  "shortest" and they all use it;
- cannot express "this traffic is a video call and that traffic is a backup".

The proposition of this project is that a system which *observes* network state
continuously and *learns* from it could do better. Testing that proposition
honestly is the whole exercise.

## 2. Routing theory you need

### 2.1 Shortest paths and additive costs

Give every edge a non-negative cost `c(e)`. The cost of a path is the sum of its
edges' costs. **Dijkstra's algorithm** finds the minimum-cost path from a source
to every other node in `O((V + E) log V)` with a binary heap.

The key property, and the one that shapes this entire project:

> **Dijkstra is optimal for any additive, non-negative edge cost.**

Not "good". Optimal. If your objective is a sum over edges, there is no algorithm
— classical, heuristic, learned, or yet to be invented — that finds a better path
than Dijkstra does. The only thing another method can do is find the same path
more slowly, or a worse path.

This is not a limitation of machine learning. It is a theorem about the problem.

### 2.2 Where the theorem stops applying

Dijkstra's optimality requires the objective to decompose as a sum over
independent edges. It stops applying when:

**(a) The objective is not additive.** "Minimise the *maximum* link utilization
along the path" is a bottleneck (min-max) objective. Standard Dijkstra cannot
express it. (A variant, the widest-path algorithm, can — but it optimises the
bottleneck *instead of* the sum, not both.)

**(b) There are constraints as well as an objective.** "Minimise latency
**subject to** total packet loss below 1% **and** no link above 70% utilization"
is a *multi-constrained optimal path* (MCOP) problem. With two or more
independent additive constraints it is NP-hard in general. Dijkstra optimises the
objective and ignores the constraints entirely — it will happily hand you the
cheapest path straight through a saturated link.

**(c) The costs depend on your own decisions.** If routing a flow over a link
raises that link's cost, then the "optimal" path for the next flow depends on
what you did with the last one. Greedy per-request shortest-path is then
demonstrably *sub-optimal* over a sequence: it self-congests. This is a sequential
decision problem, which is exactly what reinforcement learning is for.

**(d) The state is only partially observable, or must be predicted.** If you must
choose a route now for traffic that arrives in five seconds, you need a forecast,
and forecasting is a learning problem.

Every one of these is a place where a learned method has genuine room to win.
None of them existed in the original version of this project, which is why the
original benchmark showed the AI losing in all five scenarios by 30–80%.

### 2.3 The candidate-set framing

All the learned methods here operate on a *candidate set*: given a demand, we
enumerate the `k = 5` shortest loopless paths (Yen's algorithm, via networkx's
`shortest_simple_paths`) under the congestion-adjusted cost, and the policy picks
one.

This has an important consequence that must be stated up front. **A policy
restricted to a candidate set generated by shortest-path search cannot beat the
best member of that set.** Its ceiling is the *oracle over the candidate set*, not
the true optimum over all paths. We measure against exactly that ceiling, and say
so.

Why do it this way at all? Because it makes the action space small and discrete
(5 options rather than "any path in the graph"), which is what makes the learning
problem tractable at this scale. The alternative — hop-by-hop next-hop selection
over the raw graph — is what our multi-agent formulation does, and it is
correspondingly harder to train.

## 3. Why "AI beats Dijkstra" is harder than it sounds

Suppose you set out to show a neural network beating Dijkstra at routing. Here is
the trap, and it is a subtle one, because every step seems reasonable:

1. You build a network simulator. Link utilization varies over time — say, a
   random walk, because that is easy and looks realistic.
2. You define the cost of a link as a function of its latency and utilization.
3. You train a model to pick good paths under that cost.
4. You benchmark it against Dijkstra using the same cost.
5. It loses. Every time. In every scenario. By a lot.

The mistake is at step 1, and it is invisible from steps 2–5. If utilization
evolves *independently of routing*, then at every instant the problem is a static
additive shortest-path problem, which Dijkstra solves exactly. Your model's best
possible outcome is a tie. Any deviation is a loss.

You cannot fix this with a better architecture, more training, or a cleverer
reward. **The environment has to change.** Either the network must react to
routing decisions, or the objective must stop being a single additive sum.

This project made both changes, and the difference in what the experiment can
even *detect* is the single most useful thing in this document.

> **Transferable lesson.** Before spending compute on a model, ask: *given my
> environment and my objective, what is the theoretically best achievable
> performance, and what achieves it?* If the answer is "a classical algorithm you
> already have", you have not designed an experiment — you have designed a
> demonstration that the classical algorithm works.

---

# Part II — The system

## 4. Architecture

Five layers, with dependencies pointing strictly inward.

```
web/          React dashboard
service/      FastAPI: REST + WebSocket, database, experiment sandbox
experiments/  Benchmark harness — deliberately NOT part of the service
ml/           Architectures, environments, training, evaluation, checkpoints
routing/      Eight algorithms behind one Router interface
core/         Domain: models, cost, paths, QoS, simulator, network sources
```

`core/` imports nothing from the layers above it. That is what makes it the
*scientific core*: you can `import core` in a notebook and study the simulator,
the cost model and the path algebra without starting a web server. In the
original layout it lived at `backend/simulator/`, which made it look like an
implementation detail of a web application. It is the opposite: the web
application is a viewer for it.

### 4.1 The `NetworkSource` abstraction

Everything above `core/` consumes a `NetworkState` — a list of nodes, a list of
links with their current metrics, a timestamp, a step counter. It does not care
where that came from. Three implementations exist:

| Source | Closed loop? | Used for |
|---|---|---|
| `SimulatedSource` | **Yes** | every published benchmark number |
| `TraceReplaySource` | No | deterministic replay of a recorded measurement run |
| `LiveProbeSource` | No | measuring the operator's own network, live |

The distinction that matters is **closed loop**. Only the simulator can react to
our routing decisions; a recording and a real external network cannot. The API
reports this per-source (`GET /network/source` returns `closed_loop: true|false`)
because it changes what a result means.

## 5. The simulator

`core/simulator.py`. Three design decisions define it.

### 5.1 Topology: small-world, not a ring

A generator that produces a ring is useless for routing research, because between
any two nodes there are exactly two paths and no algorithm can differentiate.
This is not hypothetical — the original generator capped edges at
`min(50, n(n-1)/2)`, so at 100 nodes the ring alone supplied all 100 edges and no
extras were added. Degree 2, diameter 50. Every algorithm scored within 1% of
every other, and the scenario was labelled a stress test.

The generator now:

1. builds a ring (guarantees connectivity),
2. adds `i → i+2` shortcuts (gives *small-world* structure — short average path
   length with high clustering, as in Watts–Strogatz),
3. fills the remaining budget with random long-range links.

`target_edges = n × avg_degree / 2`, with `avg_degree = 4`.

| Nodes | Edges | Avg degree | Diameter |
|---|---|---|---|
| 25 | 50 | 4.0 | 5–6 |
| 50 | 100 | 4.0 | 7 |
| 100 | 200 | 4.0 | **8** (was 50) |

The statistics are logged at construction, so a degenerate topology can never
again go unnoticed.

### 5.2 Traffic dynamics: AR(1) around a diurnal cycle

Utilization evolves as

```
offered(link, t) = 0.30 + 0.18·sin(2π·t/40 + φ_link) + bias_link + flow_load(link, t)
u(link, t+1)     = 0.85·u(link, t) + 0.15·offered(link, t) + N(0, 0.03)
```

clipped to [0, 1]. In words: each link has a preferred load level that cycles
with a 40-step period, offset per link so they do not all peak together, and its
actual utilization *relaxes toward* that level rather than jumping to it.

Why not the original random walk `u ← u + N(0, 0.05)`? Because of a fact worth
internalising:

> **The Bayes-optimal one-step predictor for a random walk is the identity
> function.**

If your process is a random walk, the best possible forecast of the next value is
the current value. Training a 66,000-parameter LSTM to predict it means training
it to learn `f(x) = x`. Whatever low error you report is that, not intelligence —
and there is no possible experiment that distinguishes a good model from a bad
one, because the ceiling is persistence and everyone hits it.

An AR(1) process around a deterministic periodic component has *learnable
structure*: a model that identifies the period and the mean-reversion coefficient
can genuinely beat persistence. Persistence remains a strong baseline, which is
the point — it should be hard, not impossible.

### 5.3 Closing the loop

```python
def register_flow(self, path, demand=1.0):
    for i in range(len(path) - 1):
        key = self._edge_key(path[i], path[i + 1])
        self.flow_load[key] += self.load_per_flow * demand
```

Routing a flow adds load to the links it uses. Load decays exponentially
(`×0.90` per tick) so congestion is transient. Background traffic routes a few
random demands greedily each tick, so the network carries load even when the API
is idle.

This is the change that makes the project's question answerable. Consider two
policies facing 60 identical demands between the same pair:

- **Greedy**: always take the cheapest path. All 60 flows land on the same links.
- **Round-robin**: alternate across three candidate paths.

Under the old open-loop simulator these are indistinguishable — neither affects
anything. Under the closed loop, greedy saturates its chosen path while
round-robin spreads the load. This is asserted as a test
(`test_load_balancing_beats_greedy_under_repeated_demand`), and it fails if the
loop ever reopens.

**A calibration lesson.** The first implementation added the flow term as a shock
*outside* the AR(1) update:

```python
u ← a·u + (1-a)·baseline + noise + flow_load     # WRONG
```

The steady state of that recursion is `baseline + flow_load/(1−a)`, which with
`a = 0.85` amplifies the flow term **ten-fold**. Mean utilization pinned at 0.68
with p95 at 1.0 — everything saturated, no dynamic range, and the cost function
degenerate again. Folding the flow term into the offered-load baseline instead:

```python
u ← a·u + (1-a)·(baseline + flow_load) + noise   # RIGHT
```

gives steady state `baseline + flow_load`, a 1:1 mapping. Mean utilization now
sits near 0.39. **If you add a term to a recursive process, work out its steady
state before trusting the magnitude.**

## 6. The cost model

```python
CONGESTION_PENALTY_FACTOR = 4.0
CONGESTION_EXPONENT = 2

def link_cost(link):
    return link.base_latency * (1 + 4 * link.utilization ** 2)
```

An idle link costs its base latency; a saturated link costs 5×. The quadratic is
deliberate: **convexity is what makes spreading load rational.**

Two half-full links cost `2 × L × (1 + 4×0.25) = 4L`. One full link plus one idle
link costs `L × 5 + L × 1 = 6L`. Same total traffic, different arrangement,
different cost — and the convex penalty is what creates that difference. Under a
*linear* penalty the two would tie and load balancing would be pointless.

This function lives in exactly one place. It previously existed at **14 sites
across 11 files**. That is not a tidiness complaint: it defines what "good
routing" means in this system, and a partial edit would have silently made the
benchmark compare algorithms optimising different objectives — an error that
produces plausible numbers and no error message.

## 7. Quality of service

This is where the problem becomes genuinely hard, and it is the most conceptually
interesting part of the system.

### 7.1 Five classes

| Class | Optimises | Hard constraints |
|---|---|---|
| **Emergency** | loss (0.60) > latency (0.25) > utilization (0.15) | path loss ≤ 1%, bottleneck ≤ 70%, ≤ 8 hops |
| **Voice / video** | latency (0.50), loss (0.25), utilization (0.25) | loss ≤ 2%, bottleneck ≤ 80%, ≤ 8 hops |
| **Gaming** | latency (0.80) | loss ≤ 4%, bottleneck ≤ 85%, ≤ 6 hops |
| **Bulk** | utilization (0.80) — steered onto idle links | bottleneck ≤ 95% |
| **Best effort** | latency (1.00) | none |

Best-effort reproduces the project's original objective exactly, so every
historical comparison remains meaningful.

### 7.2 Why re-weighting alone would not have been enough

Here is a trap worth understanding, because it is easy to fall into.

Suppose QoS meant only "different weights per class". Then the emergency
objective is still `Σ_edges (0.25·latency + 0.60·loss + 0.15·util)` — still a sum
over independent edges. **Dijkstra solves that exactly too.** You would have
built five problems instead of one, and Dijkstra would win all five.

The constraints are what change the complexity class:

```
minimise    Σ  qos_cost(e)
subject to  Σ  loss(e)          ≤  max_path_loss        (additive constraint)
            max util(e)         ≤  max_bottleneck       (NON-additive)
            |path|              ≤  max_hops
```

The bottleneck constraint is not a sum. Multi-constrained path selection with two
or more independent constraints is NP-hard in general, and constraint-blind
Dijkstra can and does return infeasible paths — it optimises the objective and
has no way to express "but not through there".

So the headline QoS metric is **constraint satisfaction rate**, not latency. A
path that is 5 ms faster but violates the emergency loss budget is not a better
answer; it is a wrong answer.

### 7.3 One model, five classes

Rather than train five models, the class is an **input**. `profile_vector()`
encodes a profile as six numbers (three weights, three normalised constraint
levels), which is appended to the GNN's path features and to the PPO observation.
Unconstrained axes encode as 1.0, the correct saturating value for an upper
bound.

This is a small design choice with a large payoff: the model can *generalise
across classes* rather than memorising five separate mappings, and adding a sixth
class does not require retraining from scratch.

### 7.4 The honest-comparison problem

Adding QoS creates an obvious temptation: compare the learned routers against
constraint-blind Dijkstra, watch them win, and declare victory. That would be a
strawman. It is trivial to beat an algorithm at a job it was never asked to do.

So we implemented **`ConstrainedRouter`** — enumerate the *k* candidates, discard
the infeasible ones, return the cheapest survivor. This is the standard classical
approach to MCOP and it is *exact over the candidate set*. Within that set it
**is** the oracle, and a learned policy cannot beat it by construction.

That is fine, and reporting it is the point:

- `dijkstra` — the constraint-blind floor;
- `constrained` — the constraint-aware ceiling;
- `gnn`, `rl`, `multi_agent` — scored on how much of the gap they close, using
  the same candidate set all three see.

And critically: **the learned routers get no post-hoc constraint filtering.** If
the GNN returns an infeasible path, that counts as a miss. Filtering its output
through the QoS oracle would make its satisfaction rate identical to the
constrained baseline by definition, which would be a meaningless number dressed
up as a result.

---

# Part III — The algorithms

## 8. Dijkstra

**Idea.** Grow a set of nodes whose shortest distance from the source is known.
Repeatedly take the closest unknown node, finalise it, and relax its edges.

```python
distances = {node: inf for node in nodes}; distances[src] = 0
heap = [(0.0, src)]
while heap:
    cost, current = heappop(heap)
    if current in visited: continue
    visited.add(current)
    if current == dst: break
    for neighbour, weight in adjacency[current]:
        if cost + weight < distances[neighbour]:
            distances[neighbour] = cost + weight
            previous[neighbour] = current
            heappush(heap, (distances[neighbour], neighbour))
```

**Why it works.** Because costs are non-negative, the closest unfinalised node
cannot later be reached more cheaply through a node that is further away. That
"greedy choice property" is what makes the greedy algorithm optimal, and it is
also exactly what fails if a cost can be negative.

**Complexity.** `O((V + E) log V)`.

**Its one structural weakness**, which the QoS scenarios exist to expose: it
optimises an additive objective and **cannot express a constraint**.

## 9. Bellman-Ford

**Idea.** Relax every edge, `V−1` times. After `k` rounds, all shortest paths
using at most `k` edges are correct; a simple path uses at most `V−1` edges.

**Complexity.** `O(V·E)` — slower than Dijkstra.

**Why keep it?** Two reasons, one of which is honest and one of which is a
lesson.

The honest reason: it tolerates negative edge weights and detects negative
cycles, and its structure — repeated neighbour-to-neighbour relaxation — is the
basis of real distance-vector protocols like RIP.

The lesson: **it is not an independent baseline, and presenting it as one is
misleading.** With identical non-negative weights, Dijkstra and Bellman-Ford are
both exact, so they necessarily return the same cost. The original benchmark
reported `dijkstra_match_rate = 1.000` for Bellman-Ford in all five scenarios and
counted it as one of "eight algorithms compared". It is one algorithm reported
twice. It is now labelled a correctness cross-check on Dijkstra, and it is
excluded from the degeneracy guardrail for exactly that reason.

## 10. Constrained k-shortest paths

**Idea.** Enumerate the *k* cheapest loopless paths (Yen's algorithm), evaluate
each against the traffic class's constraints, return the cheapest feasible one.
If none is feasible, return the least-bad.

**Why it matters.** It is the honest classical answer to the QoS problem and the
ceiling every learned method is measured against. See §7.4.

**Complexity.** `k` × the cost of a shortest-path computation, plus `O(k·|path|)`
to evaluate. In practice ~10× Dijkstra.

## 11. Ant Colony Optimization

**Idea.** Simulate ants. Each walks from source to destination, choosing the next
hop probabilistically:

```
P(next = j)  ∝  τ(i,j)^α  ×  (1/cost(i,j))^β
```

where `τ` is *pheromone* on the edge. After every iteration, pheromone evaporates
(`×(1−ρ)`) and each ant deposits `Q/cost(path)` on the edges it used. Good paths
accumulate pheromone and attract more ants; evaporation stops the colony locking
onto a route that has since become congested.

**Parameters here.** α=1, β=2, ρ=0.2, Q=100, 20 ants × 30 iterations = **600 ant
walks per routing decision.**

**What it is good at.** Genuine multi-path exploration. Its `diversity_index` is
by far the highest of any algorithm — where every other method converges on one
path per demand, ACO keeps sampling alternatives. That is the property you want
for load spreading.

**What it costs.** ~21 ms per decision against ~2 ms for everything else — an
order of magnitude, and that wall-clock cost is *not* reflected in the latency
metric, which measures the path rather than the time taken to find it. In a
router forwarding millions of packets per second, 600 stochastic walks per
decision is not a deployable design; it is a design for computing a *routing
table* offline.

**QoS integration.** Ants are scored on the class objective with a large penalty
for constraint violation, so any feasible path outranks any infeasible one while
the search still gradients toward "less badly violating" when nothing is
feasible.

## 12. Graph neural network

### 12.1 What message passing actually is

A graph neural network learns node representations by repeatedly aggregating
information from neighbours. One round:

```
m_{i→j} = MLP_msg( h_i ‖ h_j ‖ e_{ij} )          # a message per directed edge
h'_j    = MLP_upd( h_j ‖ mean_{i∈N(j)} m_{i→j} ) # update from the mean message
```

After `L` rounds, a node's representation depends on its `L`-hop neighbourhood.
We use `L = 2`, hidden dimension 64, 47,361 parameters.

We implemented this from scratch rather than importing PyTorch Geometric. That
was a deliberate call: PyG's install matrix is a real deployment liability, and a
readable 40-line implementation demonstrates what message passing *is* rather
than how to call it.

### 12.2 Two representational failures worth studying

Both were in the original implementation, both are instructive, and neither
causes an error message.

**Failure 1: unnormalised aggregation.** The original used
`agg.index_add_(0, dst, messages)` — a *sum*. A node with four neighbours
receives four messages summed; a node with one receives one. Embedding magnitude
therefore scales with degree.

The model was trained on a topology that was accidentally a degree-2 ring, and
served on a degree-4 mesh. Activations at inference were roughly **double** what
the model saw in training. That is textbook covariate shift, self-inflicted, and
completely invisible: the model runs, produces numbers, and is quietly wrong.

Fix: divide by in-degree. Mean aggregation makes representations comparable
across topologies of different density.

**Failure 2: pooling that discards the answer.** The original path scorer was:

```python
path_embedding = h[path_nodes].mean(dim=0)
```

A mean over node embeddings is:

- **permutation invariant** — `A→B→C` scores identically to `A→C→B`;
- **length invariant** — a 2-hop and a 7-hop path through similar nodes look alike;
- **edge blind** — the utilization of the links *on the path* is never fed in
  directly, only diffused through two rounds of neighbourhood aggregation.

The model was being asked to predict a path's cost while being shown neither the
path's length nor its links. No amount of training fixes that; the information is
not in the input.

Fix: pool over **edges** as well as nodes, and concatenate explicit path features:

```python
features = [mean node embedding ‖ mean edge embedding ‖ hops, mean util, max util ‖ QoS vector]
score    = MLP(features)
```

> **Transferable lesson.** When a model underperforms, before touching the
> architecture or the hyperparameters, ask: *is the information required to
> answer the question present in the input at all?* Ours was not.

### 12.3 Training it as a ranker

The model's output is consumed by exactly one operation: `argmin`. The absolute
score is never used.

Training it with MSE against the true cost therefore optimises the wrong thing —
it spends capacity getting the *magnitude* right, which nobody reads, and is not
directly penalised for getting the *ordering* wrong, which is all that matters.

We train with a pairwise margin ranking loss:

```python
sign      = torch.sign(target_j - target_i)          # +1 when j is genuinely worse
violation = torch.relu(margin - sign * (pred_j - pred_i))
loss      = violation[off_diagonal].mean()
```

and report **top-1 accuracy** and **mean regret against the oracle** — the two
numbers that describe how the model is actually used.

> **Transferable lesson.** Train the thing you use. If you only consume `argmin`,
> optimise ordering, not values.

### 12.4 The validation set that was not one

The original generated training data from a simulator, then generated validation
data by **continuing the same simulator instance** — same topology, same seed,
same base latencies, just later timesteps. The reported "72% validation MSE
reduction" measured how well the model interpolated within one trajectory. It was
not evidence of generalisation to anything.

Now: three independently seeded simulators for train, validation and test. That
one change is why the current number (top-1 0.978 on held-out data against 0.227
random) means something.

## 13. Reinforcement learning

### 13.1 The formulation

- **State** — the 286-dimensional observation below.
- **Action** — `Discrete(5)`, an index into the candidate set.
- **Reward** — negative class-weighted path cost, minus an infeasibility penalty,
  minus a global load-balance term measured *after* the chosen flow lands.
- **Episode** — 200 decisions on a freshly seeded simulator.
- **Algorithm** — PPO, 250,000 timesteps.

### 13.2 The unobservable MDP

This is the most instructive bug in the entire project.

The original observation encoded **per-link features only** — utilization, queue,
loss, latency for each of 50 links. Meanwhile `_sample_routing_task()` re-drew
`(source, destination)` **every single step**.

Think about what the agent sees. It is handed 200 numbers describing link states
and asked to output an integer in [0, 5). It is never told which pair it is
routing. And because the pair changes every step, the *meaning* of "action 2"
changes every step: sometimes it is a 3-hop path, sometimes 6 hops, sometimes a
completely different region of the network.

This is not a partially observable MDP, where the agent has incomplete
information about a consistent problem. It is an **unobservable** one: no
mapping from observation to action can do better than the marginal best index,
because the observation carries no information about which decision is being
made.

The evidence was in the repository the whole time. `evaluations.npz`:

| | |
|---|---|
| First evaluation (25k steps) | −86.57 |
| Last evaluation (500k steps) | −91.45 |
| Best checkpoint | −86.57, **the first one taken** |
| OLS slope | −0.094 per 100k steps |
| r² | **0.001** |
| p | **0.878** |

Five hundred thousand timesteps of training produced a curve statistically
indistinguishable from a flat line. Meanwhile the documentation asserted "mean
reward improved from −77 to −61 (+21%)" with "best −45.81" — figures that appear
nowhere in the file.

That last part is the part to sit with. Nobody wrote those numbers dishonestly;
they came from a run that was never committed. But **nothing in the system could
notice**, so the claim and the evidence drifted apart and stayed apart. That is
why `scripts/verify_claims.py` now exists.

### 13.3 The fixed observation

| Block | Width | Contents |
|---|---|---|
| Link state | 50 × 4 = 200 | utilization, queue, loss, base latency |
| **The task** | 25 × 2 = 50 | one-hot source, one-hot destination |
| **The choices** | 5 × 6 = 30 | per candidate: valid, hops, cost, mean util, max util, loss |
| **The class** | 6 | QoS weights and constraints |
| | **286** | |

The last three blocks are new. The agent can now see which problem it is solving
and what its options are.

### 13.4 The reward term with no gradient

The original reward included "global load balancing terms" — utilization variance
and maximum utilization computed over **all** links — with a docstring claiming
they "give RL a genuinely different objective from Dijkstra".

They were computed from the state *before* the action, over all links,
independent of which action was taken. In policy-gradient terms, a reward
component that does not depend on the action is a **state-dependent baseline**.
Its effect on the policy gradient is exactly zero. It shifts returns and adds
variance to the estimator; it cannot change the optimal policy.

The fix is one line, and it required the closed loop to exist first:

```python
if chosen:
    self._sim.register_flow(chosen)   # our flow lands
new_state = self._sim.step()          # time advances
reward = immediate + self._global_reward(new_state)   # now action-dependent
```

Because our own flow is part of what produced `new_state`, the global term now
genuinely depends on what we did.

> **Transferable lesson.** In policy gradient methods, a reward term that does
> not depend on the action contributes nothing to learning, no matter how
> important it looks in the code.

### 13.5 Reporting the result honestly

The final number is not an episode return. It is a **normalized score**:

```
normalized = (policy − random) / (oracle − random)
```

0.0 means no better than random; 1.0 means it matches the greedy oracle. Both
reference points are evaluated on the *same seeded episodes*, so the comparison
is paired.

| Policy | Mean return | Normalized |
|---|---|---|
| Random | −48.01 | 0.000 |
| Greedy cheapest (≈ Dijkstra) | −34.66 | 0.903 |
| **PPO** | **−35.20** | **0.867** |
| Greedy oracle | −33.23 | 1.000 |

**PPO learned — and still loses to the greedy baseline.** On a largely additive
objective there is only ~10% of headroom above greedy, and the policy has not
captured it.

Reporting that is not a failure of the write-up; it *is* the result. A reader who
sees only "0.867 normalized" learns that the model works. A reader who sees the
greedy row too learns the far more useful fact that on this objective there was
almost nothing to win.

## 14. Multi-agent reinforcement learning

### 14.1 What it was, and why that name was wrong

The original: N independently trained PPO agents, each receiving the **full
global** observation, each emitting a **complete end-to-end path**, with the
acting agent selected by a lookup on the source node.

Count what is missing. No joint action space. No inter-agent communication. No
credit assignment. And no decentralization of *either* observation or action —
every agent saw everything and decided everything. That is a **mixture of experts
with a hardcoded gating function**. It was documented as
"centralized-critic, decentralized-execution multi-agent RL", which is a specific
technical claim (CTDE) that the code did not support.

### 14.2 What CTDE actually means

**Centralized Training, Decentralized Execution** is a standard MARL paradigm:

- At **execution** time each agent acts on *local* information only, because at
  deployment there is no global oracle — a router knows its own link states, not
  the whole internet's.
- At **training** time the *critic* may use global information, because training
  happens offline where you do have it. A better value estimate reduces gradient
  variance without contaminating the policy.

The asymmetry is the entire idea. If the actor sees global state, it is not
decentralized. If the critic does not, it is not centralized training.

### 14.3 The implementation

**Decentralized execution.** Observation, 113 dimensions:

```
[ current node one-hot within region      (40) ]
[ destination: is-here, hop distance, same-region (3) ]
[ 8 neighbours × 8 features               (64) ]   ← utilization, queue, loss,
[ QoS class                                (6) ]      latency, is-destination,
                                                      same-region, gets-closer
```

**Its width is a constant** — 113 whether the network has 25 nodes or 100. This
is the property that makes "local" a fact rather than a claim, and it is asserted
by a test that instantiates both and compares shapes.

Action: `Discrete(8)` — one neighbour, not a path. A route emerges from several
agents acting in sequence as the packet crosses regions.

**Centralized training.** A 16-dimensional global summary — a utilization
histogram plus moments — is appended for the critic. Deliberately a *summary*
rather than the raw link vector, because a raw vector could not transfer between
topologies.

The asymmetry is enforced by two feature extractors:

```python
class AsymmetricExtractor(BaseFeaturesExtractor):
    def forward(self, obs):
        view = obs if self.use_global else obs[:, :self.local_dim]
        return self.net(view)
```

Actor: `use_global=False`. Critic: `use_global=True`.

**Verified, not asserted.** A test perturbs *only* the global block and checks
that the action distribution does not move while the value estimate does. If
someone accidentally gives the actor global access, that test fails.

### 14.4 A bug about serialization worth remembering

The asymmetry was first applied as a patch *after* constructing the policy:
build the PPO, then replace `policy.vf_features_extractor`.

That works at training time and breaks silently at load time. `PPO.load` rebuilds
the policy by calling its constructor with the saved `policy_kwargs`, and SB3
hands **both** extractors the same kwargs. The patch was never re-applied, the
rebuilt critic had the wrong input width, and `load_state_dict` failed.

The failure was caught — and then swallowed by the router's `try_load_models`,
which logged a warning and fell back. The result: an entire benchmark run where
`multi_agent` reported `fallback_rate = 1.00`. The guardrail worked exactly as
designed; it told us the row was not AI.

The fix is to put the asymmetry in the **class**, so reconstruction matches
construction by definition.

> **Transferable lesson.** If you modify a model after construction, ask what
> happens when the framework reconstructs it. Anything not expressed in the
> constructor or the saved config does not survive a round trip.

### 14.5 What we do and do not claim

Claimed and verified: decentralized execution, local observations of constant
width, next-hop actions, control transfer between regions, a centralized critic,
partitioning derived from the live topology.

**Not claimed:** no explicit inter-agent messaging; no shared critic across
agents. These are *independent learners* trained in rotation against frozen
partners, cooperating through the shared environment and a shared team reward
term (the change in network-wide utilization variance). That is a real and
standard MARL setting. It is not a joint-action solver, and the model card says
so in those words.

## 15. LSTM congestion forecasting

### 15.1 The task

Given the last 20 utilization vectors, predict the next one. Route on the
prediction instead of the present, and you steer around congestion that has not
arrived yet.

### 15.2 Why the first attempt scored −1.77

Trained to predict the utilization **level**, the model achieved test MSE
0.001137 against persistence's 0.001337 — wait, that is *better*. The first
attempt scored 0.003705 against persistence's 0.001337.

**Skill score = 1 − MSE_model / MSE_persistence = −1.77.** Nearly three times
worse than copying the last value forward. The training script refused to save
the checkpoint, which is the behaviour we wanted: a forecaster that cannot beat
persistence should not be deployed.

The diagnosis. Utilization is strongly autocorrelated (`a = 0.85`), so
`u_{t+1} ≈ u_t` is right to within the one-step noise almost every time. A network
asked to emit the level must first reconstruct the input — learn the identity —
and only then add the small correction. Almost all of its capacity goes to the
part that is free.

### 15.3 Predicting the residual

```python
def delta(self, inputs):
    differenced = inputs[:, 1:, :] - inputs[:, :-1, :]   # stationary input
    out, _ = self.lstm(differenced)
    return MAX_DELTA * torch.tanh(self.output(out[:, -1, :]))

def forward(self, inputs):
    return torch.clamp(inputs[:, -1, :] + self.delta(inputs), 0, 1)
```

The loss is computed on `u_{t+1} − u_t`. Persistence becomes the specific
hypothesis "the residual is zero", which is a fair fight, and every unit of
capacity goes to the diurnal cycle and the mean-reversion drift.

| Predictor | Test MSE |
|---|---|
| Window mean | 0.016811 |
| Persistence | 0.001337 |
| **LSTM (residual)** | **0.001137** |

**Skill score +0.1497.** Saved.

A 15% reduction in one-step error is a modest result, and the modesty is honest:
on an AR(1) process with σ = 0.03 noise, persistence is close to optimal and there
is not much left to win.

> **Transferable lesson.** Differencing is the standard first move for a strongly
> autocorrelated series, and the reason is not statistical folklore — it is that
> predicting levels forces the model to spend capacity relearning the identity.

### 15.4 The feature that had never executed

Before this, predictive routing had **never once run**. The artifact was absent,
so `build_forecast_state` returned `None` on every call and the caller fell
through to `or state` — routing on the present while labelling it a forecast.

The observable symptom was in every committed result file: `gnn_predictive` was
**byte-identical** to `gnn`, and `rl_predictive` to `rl`. Two of the eight
benchmarked "algorithms" were duplicate columns.

This is now a permanent honesty gate: if the predictive variants ever equal their
base algorithms again, CI fails.

---

*Part IV (methodology and results) and Part V (applications and deployment)
continue below.*
