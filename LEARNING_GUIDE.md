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

# Part IV — Method and results

The three chapters in this part are, in order: how the experiment is
constructed, what it produced, and what the numbers are allowed to mean. They
are separated deliberately. A great many project reports fuse them, and the
fusion is where overclaiming hides — a result stated inside a paragraph of
interpretation is very hard to check.

---

## 16. Experimental methodology

### 16.1 The unit of replication

This is the single most important paragraph in Part IV, so it goes first.

The original benchmark reported p-values in the range `1e-40` and one that was
literally `0.0`. It obtained them like this: run one simulation, ask each
algorithm to route 20,000 source–destination pairs as the simulation evolved,
collect 20,000 latencies per algorithm, and run a t-test on the two vectors.

Twenty thousand numbers is a large sample only if the numbers are independent.
These were not remotely independent:

- Decision *t* and decision *t+1* see almost the same network. Utilisation is an
  AR(1) process with `a = 0.85`, so consecutive observations share about 85% of
  their signal.
- Every decision in the run shares one topology draw, one traffic seed, one
  diurnal phase.
- Once the loop was closed, decision *t* **caused** part of the state at
  decision *t+1*.

The effective sample size was therefore not 20,000. It was closer to **one**:
one run. A t-test fed 20,000 correlated observations does not become more
correct as *n* grows — it becomes more confidently wrong, because the standard
error shrinks like `1/sqrt(n)` on a denominator that was never valid. This
failure mode has a name in the experimental-design literature:
**pseudo-replication**, described by Hurlbert in 1984 for ecology field studies,
and it is exactly the same mistake.

The fix is to be explicit about the unit of replication and then never violate
it:

> **The unit of replication is one independently seeded run.**

The protocol that follows from that sentence:

1. Choose `n_runs = 15` seeds (`base_seed = 1000`, then 1001, 1002, …).
2. For each seed, build the topology, the traffic process and the demand
   schedule **once**. Every algorithm gets that same seed's world.
3. Each algorithm runs its **own closed-loop trajectory** through that world
   (see 16.2), producing `n_steps × m_pairs = 40 × 8 = 320` decisions.
4. Collapse those 320 decisions into **one number per algorithm per seed** — the
   run's mean latency.
5. Statistics are computed across the 15 run-level numbers. Never across the 320.

The sample size in every reported test is therefore `n = 15`, and it is printed
in the results file next to the p-value. Fifteen is a small sample. It is also
an honest one, and the paired design (16.5) buys back most of the power that
the smaller *n* costs.

Every result file carries this block, so a reader never has to reconstruct it:

```json
"replication": {
  "n_runs": 15, "n_steps": 40, "m_pairs": 8, "base_seed": 1000,
  "unit_of_replication": "one independently seeded run",
  "note": "Each algorithm runs its own closed-loop trajectory per seed with an
           identical topology, background traffic and demand schedule.
           Statistics are computed across runs, never across the autocorrelated
           decisions within one run."
}
```

### 16.2 Per-algorithm closed-loop trajectories

Section 5.3 explained why the simulator has to be closed-loop: if routing
decisions do not change utilisation, then congestion-aware routing has nothing
to be aware of, and every algorithm is scored against a movie it cannot affect.

Closing the loop creates a measurement problem the open-loop design did not
have. If all eight algorithms route through **one shared** simulator, they
pollute each other's environment: ACO's long, wasteful paths load links that
Dijkstra then has to avoid, so Dijkstra's score depends on which competitors
happened to be in the room. That is not a benchmark, it is a food fight.

So each algorithm gets **its own simulator instance**, constructed from the same
seed. Eight identical worlds, eight independent trajectories, one comparison per
seed. Algorithm *A*'s congestion is caused only by algorithm *A*.

The cost of this design is that a "did these two algorithms pick the same path?"
metric stops being meaningful — by step 5 they are looking at genuinely
different networks, so disagreement no longer implies a different *policy*. That
is why degeneracy needs its own probe (16.7).

### 16.3 Scenarios

Seven declarative scenarios, defined in `experiments/scenarios.py`. Each is a
small class that can perturb the simulator at construction time (`prepare`) and
at every tick (`on_step`), so a scenario is data, not a fork of the runner.

| Scenario | What it tests |
|---|---|
| `normal_traffic` | The baseline. Simulator dynamics only. |
| `high_congestion` | Elevated background load; the regime where cost differences should show. |
| `link_failures_persistent` | Links removed at the start and never restored. Tests routing on a damaged graph. |
| `cascading_failure` | Failures introduced progressively during the run; load displaced onto neighbours. |
| `congestion_bursts` | Sharp transient hot-spots. This is the scenario a forecaster should win. |
| `large_topology_100_nodes` | Scale. 100 nodes, ~200 edges. Tests whether anything degrades or times out. |
| `qos_mixed_traffic` | Five traffic classes with hard constraints. The regime where learned routing has room to win. |

The last two exist because of specific arguments made in Part III. Section 12.2
predicted that a fixed-width model would have trouble at a different topology
size; `large_topology_100_nodes` is the test that makes that prediction
falsifiable. Section 7 argued that additive-cost routing is where Dijkstra is
unbeatable; `qos_mixed_traffic` is the scenario built to leave that regime.

### 16.4 Metrics

Per algorithm, per scenario:

| Metric | Definition | Why it is here |
|---|---|---|
| `mean_latency` | Mean realised path latency over all decisions. | The headline. |
| `p95_latency` | 95th percentile. | Means hide tail behaviour, and tails are what users feel. |
| `success_rate` | Fraction of requests that produced a path. | A fast router that fails is not fast. |
| `qos_satisfaction_rate` | Fraction of paths meeting the class's hard constraints. | The only metric that can distinguish QoS routing. |
| `fallback_rate` | **Fraction of decisions produced by the heuristic fallback rather than the model.** | See below. |
| `mean_path_max_utilization` | Mean over decisions of the most-loaded link on the chosen path. | Bottleneck pressure; a non-additive quantity. |
| `diversity_index` | Shannon entropy over the distribution of chosen paths, normalised. | Detects a router that has collapsed onto one path. |
| `mean_hops` | Mean path length. | Cheap sanity check; a router at 4.2 hops where Dijkstra is at 2.7 is doing something structurally different. |
| `dijkstra_match_rate` | From the degeneracy probe (16.7), not the main trajectories. | Detects a model that has learned to be Dijkstra. |

`fallback_rate` deserves its own note. Every learned router in this system can
fail to produce a decision — a checkpoint is missing, an observation width does
not match, a forward pass raises. In the original code those cases silently
returned the heuristic's answer and the row was still labelled "GNN". A model
that never loaded therefore reported a plausible latency and nobody could tell.
Now every `RouteResult` carries `is_fallback`, the runner aggregates it, and any
algorithm above 20% fallback gets a warning emitted into the results file naming
it. `link_failures_persistent` shows this working in the live data: `rl` scores
`fallback_rate = 1.00` and the file says, in text,

> `rl: 100% of decisions came from the heuristic fallback, not a trained model.
> This row is a heuristic, not rl.`

That is the system telling the truth about itself under conditions its author
did not anticipate. It is worth more than any of the latency numbers.

### 16.5 Statistical tests

Three quantities are reported for every algorithm against Dijkstra, computed on
the 15 paired run-level means.

**Wilcoxon signed-rank test** rather than a paired t-test. The distribution of
per-run mean latency is right-skewed (a run that happens to draw a congested
diurnal phase produces a long tail), *n* is 15, and normality is neither
observed nor testable at that sample size. Wilcoxon assumes only that the
paired differences are symmetric about their median, and pairing is exactly
what the design gives us: run *i* under algorithm *A* and run *i* under
algorithm *B* share a topology, a traffic seed and a demand schedule, so the
difference cancels the run-to-run variance that dominates the raw numbers.

**Cliff's delta** as the effect size. `d` is the probability that a randomly
chosen value from one group exceeds a randomly chosen value from the other,
minus the reverse: `d = (#(x > y) − #(x < y)) / (n·m)`. It ranges over `[−1, 1]`,
it is non-parametric, and — this is why it is here rather than Cohen's *d* — it
does not assume equal variances or normality. Thresholds reported alongside it
follow Romano et al.: `|d| < 0.147` negligible, `< 0.33` small, `< 0.474`
medium, else large.

**Bootstrap 95% confidence interval** on the mean difference, 10,000 resamples.
The CI is what a reader should actually look at. A p-value answers "could this
be zero?"; the CI answers "how big could it plausibly be?", which is the
question that decides whether anyone should care.

All three appear together, always, in `comparison_vs_dijkstra`:

```json
"comparison_vs_dijkstra": {
  "n_runs": 15, "mean_diff": 0.42, "pct_diff": 0.74,
  "wilcoxon_p_value": 0.561, "cliffs_delta": 0.031,
  "effect_magnitude": "negligible",
  "ci95_low": -0.64, "ci95_high": 1.48
}
```

Read that block as a unit. `p = 0.56` alone would tempt you to say "no
difference". The CI is the honest statement: the difference is somewhere between
0.64 ms better and 1.48 ms worse, and with 15 runs we cannot distinguish it from
zero. Those are different sentences and only the second is defensible.

### 16.6 The honest ceiling and the honest floor

A benchmark with no floor cannot tell you whether a policy learned anything, and
a benchmark with no ceiling cannot tell you whether the remaining gap is worth
chasing. Both are now first-class algorithms in the comparison table:

- **`random_baseline`** (`routing/random_baseline.py`) — picks uniformly from
  the candidate set. This is the floor. If a learned model does not beat it, the
  model is noise, and the benchmark should say so in the same table rather than
  in a footnote.
- **`constrained`** (`routing/classical/constrained.py`) — k-shortest paths
  filtered by QoS feasibility, then scored. Under additive costs this is the
  optimum; under QoS constraints it is the best a non-learned method can do
  given the same candidate set. This is the ceiling.

For the RL agent specifically, the same idea is applied at the episode level in
`ml/evaluation/baselines.py`, which reports a **normalized score**:

```
normalized = (policy_return − random_return) / (oracle_return − random_return)
```

0.0 means "no better than random"; 1.0 means "matches the greedy oracle". A raw
return of −35.2 is uninterpretable on its own. A normalized score of 0.87 is not.

### 16.7 Degeneracy detection

The most useful thing this benchmark does is refuse to let a model hide.

`experiments/runner.py` runs a separate `degeneracy_probe()`: a **shared-state,
open-loop** pass in which every algorithm is asked to route the same pairs on
exactly the same network states. Because nothing is fed back, disagreement now
means a genuine difference in policy, and `dijkstra_match_rate` becomes
meaningful again. (This is the one place open-loop is the right design, and it
is the reason the probe is separate from the main trajectories rather than
replacing them.)

Any algorithm that matches Dijkstra above 95% of the time gets a warning written
into the results file:

> `gnn: chooses the same path as Dijkstra 98% of the time, so it is degenerate
> and adds no information.`

Three algorithms are exempt — `dijkstra`, `bellman_ford` and `constrained` —
because agreeing with Dijkstra is what they are *supposed* to do; flagging
Bellman–Ford for computing shortest paths would be theatre.

The honesty gate in `tests/honesty/test_honesty_gates.py` does not require that
no algorithm is degenerate. That would be an unsatisfiable test, because the GNN
converging to Dijkstra under additive costs is the *correct* outcome (section
3). It requires that any degeneracy present is **declared** in `warnings`. The
project is allowed to have a negative result. It is not allowed to have an
undisclosed one.

### 16.8 Reproducibility

Everything in Part IV regenerates from a clean checkout:

```bash
make train        # ~35 min, CPU only, 4 models
make benchmark    # ~15 min, 7 scenarios × 8 algorithms × 15 runs
make report       # writes docs/14_RESULTS_AND_FINDINGS.md + figures
make verify       # cross-checks every documented number against the artifacts
```

Seeds are fixed (`42` for training, `1000+i` for evaluation). `make verify` is
the one that matters most: `scripts/verify_claims.py` re-reads the results files
and fails if a number written in the documentation is not present in the
artifacts. It exists because of one specific incident, described in 17.6.

---

## 17. Results

All numbers below are regenerated artifacts, not transcriptions. The canonical
copy lives in `docs/14_RESULTS_AND_FINDINGS.md` and the raw JSON in
`experiments/results/`. Where a number appears here and there, `make verify`
checks that they agree.

### 17.1 Component-level results

Each model is evaluated on its **own** task before it is allowed into the
routing comparison. This ordering is deliberate: a model that fails its own task
cannot be diagnosed from a routing table.

**GNN router** — pairwise path ranking, held-out test set:

| | Top-1 accuracy | Mean regret |
|---|---|---|
| Trained GNN | **0.978** | 0.00064 |
| Random choice | 0.227 | 0.676 |

25 epochs, 2,500 train / 600 val / 600 test samples, 47,361 parameters, 594 s on
CPU. The validation set is drawn from **independently seeded simulator runs**,
not from a random split of one run's snapshots — the original split leaked
across an 85%-autocorrelated series, so "validation accuracy" was measuring
memorisation of neighbouring timesteps (12.4).

Regret of 0.00064 means: when the model is wrong, it is wrong by six
hundred-thousandths of a cost unit. It is not merely picking the right path 98%
of the time; the 2% of the time it is wrong, it is wrong about paths that are
nearly tied.

**PPO routing agent** — 20 evaluation episodes, seed 1041:

| Policy | Mean return | 95% CI | Normalized |
|---|---|---|---|
| Random | −48.01 | [−49.40, −46.62] | 0.000 |
| **PPO** | **−35.20** | [−36.40, −34.01] | **0.866** |
| Greedy first-candidate | −34.66 | [−36.01, −33.31] | 0.903 |
| Oracle | −33.23 | [−34.29, −32.17] | 1.000 |

Read this honestly, which is how 13.5 insists it be read. PPO gets 87% of the
way from random to the oracle — that is real learning, from a genuinely
unlearnable starting point (13.2). It also **does not beat the greedy
first-candidate heuristic**, whose CI overlaps its own. The correct sentence is:
*the agent learned the task, and the task is one a one-line heuristic already
solves.* Both halves are in the results file.

The training curve, which used to be the project's headline evidence, is now
reported with its regression: slope **+0.742 reward per 100k steps**,
r² = 0.195. Before the environment fixes the same regression gave slope
**−0.094 per 100k**, r² = 0.001, p = 0.878 — a flat line through noise that the
old documentation described as a 21% improvement.

**Congestion LSTM** — chronological 70/15/15 split, 880 test windows:

| Predictor | Test MSE |
|---|---|
| Window mean | 0.01681 |
| Persistence (ŷ = last observed) | 0.00134 |
| **LSTM** | **0.00114** |

Skill score against persistence: **+0.1497**. Before predicting the residual
instead of the level (15.3), the same architecture scored **−1.77** — 2.8× worse
than doing nothing. A skill score of 0.15 is a modest but genuine result on a
0.85-autocorrelated series, and it is stated as such: the LSTM extracts about 15%
of the variance persistence leaves on the table.

**Regional CTDE policies** — 4 regions, 50k timesteps each, 2 rounds:

| Region | Nodes | Trained return | Random return | Improvement |
|---|---|---|---|---|
| 0 | 8 | 70.49 | −6.58 | +77.07 |
| 1 | 7 | 78.48 | −2.73 | +81.21 |
| 2 | 6 | 69.90 | −2.85 | +72.75 |
| 3 | 4 | 108.35 | +39.32 | +69.02 |

4 of 4 regions beat random. Each agent observes 113 local features and cannot
see outside its region; the critic sees 16 global features during training only
(14.2). This is the component result. The system-level result in 17.2 is much
less flattering, and the gap between them is itself the finding — see 18.3.

### 17.2 System-level routing results

Across the best-effort scenarios (`normal_traffic`, `high_congestion`,
`link_failures_persistent`, `cascading_failure`, `congestion_bursts`), the
picture is remarkably stable. Taking `normal_traffic` as representative
(25 nodes, 50 edges, degree 4.0, diameter 5, 15 runs, all models loaded):

| Algorithm | Mean latency | p95 | vs Dijkstra | Cliff's δ | p | Diversity | Hops |
|---|---|---|---|---|---|---|---|
| `constrained` | 56.57 | 100.93 | −0.3% | −0.004 | 0.978 | 0.260 | 2.69 |
| `gnn` | 56.71 | 101.49 | −0.0% | +0.040 | 0.679 | 0.277 | 2.68 |
| `dijkstra` | 56.73 | 100.98 | — | — | — | 0.274 | 2.70 |
| `rl` | 56.73 | 100.96 | +0.0% | +0.049 | 0.804 | 0.300 | 2.70 |
| `bellman_ford` | 57.15 | 100.72 | +0.7% | +0.031 | 0.561 | 0.272 | 2.71 |
| `random_baseline` | 84.40 | 136.32 | +48.8% | +1.000 | 6.1e-05 | 0.861 | 3.80 |
| `multi_agent` | 91.40 | 220.02 | +61.1% | +1.000 | 6.1e-05 | 0.444 | 3.67 |
| `aco` | 92.08 | 199.05 | +62.3% | +1.000 | 6.1e-05 | 0.704 | 4.24 |

Four things in that table are worth stopping on.

**(1) The top five rows are statistically indistinguishable.** Every p-value is
> 0.5, every Cliff's delta is negligible, and every bootstrap CI straddles zero.
Dijkstra, Bellman–Ford, constrained k-shortest, the GNN and the PPO agent all
route best-effort traffic to within half a millisecond of one another. This is
the central result of the project and section 18.1 is about what it means.

**(2) The gap that *is* significant is the one that should be.** Random,
multi-agent and ACO are all 48–62% worse than Dijkstra with `δ = 1.0` and
`p = 6.1e-05` — the smallest p-value a 15-pair Wilcoxon can produce. `δ = 1.0`
means *every single one of the 15 paired runs* went the same way. The test has
no trouble detecting a real difference; that is what makes its failure to detect
one among the top five informative rather than merely underpowered.

**(3) The scenario stress moves everything together.** Under `high_congestion`
Dijkstra rises 56.73 → 70.18 and the GNN rises 56.71 → 71.06; under
`link_failures_persistent`, 64.04 and 63.72. No algorithm's *ranking* changes.
Congestion makes routing harder for everyone by roughly the same amount, which
is what you would expect if all the top performers are computing the same
shortest path on the same cost function.

**(4) `diversity_index` finally means something.** It read 0.000 for every
algorithm in the original results because it was computed over a set that always
had one element. It now separates the algorithms cleanly: 0.26–0.30 for the
shortest-path family (they mostly reuse good paths), 0.44 for multi-agent,
0.70 for ACO, 0.86 for random. ACO's high diversity next to its poor latency is
the signature of a stochastic explorer that has not converged — see 18.3.

### 17.3 Failures and fallbacks under stress

`link_failures_persistent` produced the most instructive row in the entire
benchmark. `rl` reports `fallback_rate = 1.00`: **not one** of its 15 × 320
decisions came from the trained policy.

The cause is structural, not a bug. The PPO observation is a fixed-width vector
(`ml/features.py`): `links × 4 + nodes × 2 + K_PATHS × 6 + QOS_FEATS`, which is
286 for a 25-node/50-link graph. Persistent link failures *remove links*, so the
observation the environment can build no longer matches the width the
checkpoint was trained on. `_observation_fits()` detects the mismatch,
`RouteResult.is_fallback` is set, the heuristic answers, and the results file
says so in plain English.

The old code path would have silently returned the same heuristic answer under
the label "rl", and the row would have read 63.38 ms — a *better* number than
Dijkstra's 64.04 — with nothing anywhere indicating that no neural network had
been involved. Someone would have written "the RL agent outperforms Dijkstra
under link failures" in a report. That sentence would have been about a
five-line heuristic.

This is the concrete payoff of the fallback instrumentation, and it is also the
clearest statement of the fixed-width limitation discussed in 12.2 and 21.

### 17.4 The QoS regime

`qos_mixed_traffic` is the scenario built to test the argument in section 7:
that Dijkstra's optimality guarantee is a statement about *additive* costs, and
that hard per-class constraints on jitter, loss and bottleneck utilisation leave
that regime. Under multi-constrained routing the problem is NP-hard, no
polynomial algorithm computes the optimum, and a learned heuristic is competing
against other heuristics rather than against a proof.

The metric that matters here is not `mean_latency` — it is
`qos_satisfaction_rate` per class, because a router that is 3 ms faster while
violating a real-time class's jitter bound has not won anything. The comparison
is against `constrained`, which is the honest ceiling: k-shortest paths filtered
by feasibility. Beating Dijkstra on QoS satisfaction is trivial and meaningless
(Dijkstra is not trying); beating `constrained` would be a genuine result.

Here is what happened. QoS satisfaction rate, by traffic class, 15 runs:

| Algorithm | emergency | interactive | gaming | bulk | best-effort | Overall |
|---|---|---|---|---|---|---|
| **Constrained** | **92.4%** | **97.1%** | **98.1%** | **99.9%** | 100% | **97.5%** |
| GNN | 82.3% | 94.2% | 96.2% | 99.3% | 100% | 94.3% |
| Dijkstra | 84.0% | 92.7% | 95.1% | 99.2% | 100% | 94.2% |
| Bellman-Ford | 85.1% | 90.5% | 93.1% | 99.5% | 100% | 93.5% |
| RL (PPO) | 78.5% | 90.8% | 92.8% | 98.6% | 100% | 92.2% |
| ACO | 76.1% | 90.2% | 88.6% | 99.7% | 99.8% | 90.9% |
| Random | 61.7% | 80.8% | 87.0% | 96.0% | 100% | 85.4% |
| Multi-agent | 61.5% | 75.9% | 77.4% | 95.1% | 100% | 82.0% |

Four things to take from that table.

**The aggregate column is the least useful one.** The overall spread is 82.0% to
97.5% — about 15 points. The emergency spread is 61.5% to 92.4% — about 31
points. `best_effort` has no hard constraints, sits at 100% for everyone, and
pulls every row toward the same number. Reporting only the aggregate would have
halved the visible signal, and it would have hidden the class an operator
actually holds an SLA on. This is why `experiments/runner.py` records
`qos_satisfaction_rate__<class>` per run.

**The classical ceiling wins, and wins every class.** `constrained` is the top
row throughout. It is not a learned method; it is k-shortest paths plus a
feasibility filter. It also pays for it: 66.44 ms mean latency against
Dijkstra's 65.37 — about 1.6% slower. That is exactly the trade you want a
constraint-aware router to make, and it is visible only because latency and
satisfaction are reported side by side.

**The learned routers do not beat it, and the reason is not mysterious.** The
GNN edges past Dijkstra on three of five classes and loses on emergency; PPO is
worse than Dijkstra everywhere and is significantly slower (+5.9%, Cliff's
δ = 0.564, p = 0.0003). Neither model has ever been asked to satisfy a
constraint. The GNN was trained as a ranker on additive cost; PPO was rewarded
for latency. **They behave exactly as trained.** The honest description of this
result is a *training-objective gap*, not evidence that learned routing cannot
work here — and closing it (a feasibility term in the reward, a
constraint-aware ranking loss) is the single most direct experiment this
repository leaves undone.

**Multi-agent lands at random's level on the constrained classes.** 61.5% vs
random's 61.7% on emergency. The regional policies optimise local utilisation
variance and no agent can evaluate an end-to-end constraint, so under
constraints the composition is no better than chance. Consistent with 18.3, and
a sharper statement of it than the latency numbers give.

So the correct summary of the QoS work is: **the project built the arena in
which learned routing could win, and has not yet trained a model that wins in
it.** That is a smaller claim than the project's title, and it is the one the
data supports.

### 17.4.1 Scale: the 100-node scenario

`large_topology_100_nodes` (100 nodes, 200 links, degree 4.0, diameter 9)
produced the cleanest architectural result in the suite.

| Algorithm | Mean latency | Fallback rate | Matches Dijkstra |
|---|---|---|---|
| Dijkstra | 82.05 | 0% | — |
| Constrained | 82.40 | 0% | 100% |
| GNN | 82.36 | **0%** | 98% |
| RL (PPO) | 82.98 | **100%** | 100% |
| Bellman-Ford | 83.86 | 0% | 100% |
| Random | 102.89 | 0% | 19% |
| Multi-agent | 150.17 | 4% | 38% |
| ACO | 161.63 | 0% | 22% |

The PPO agent's fallback rate is **100%** — the same failure as in 17.3, from the
same cause. Its observation is 286-dimensional for a 25-node/50-link graph and a
100-node graph does not fit. Its 82.98 ms is a heuristic's number wearing the
`rl` label, and the results file says so.

The GNN runs on the larger topology with **no fallbacks at all**, because
message passing does not care how many nodes there are: the same weights apply
to a 25-node graph and a 100-node one. It was trained on 25 nodes and evaluated
on 100 without modification.

That contrast is an argument about **architecture**, not about training or
hyperparameters, and it is the most transferable single lesson in Part IV. If
you want a model that survives contact with a network whose size you did not
choose in advance, encode the graph structurally. `LEARNING_GUIDE.md` §21.2
item 3 — replacing PPO's flat observation with the GNN's node embeddings — is
the direct consequence.

ACO's degradation is also worth noting: 8.60 mean hops against an optimum of
4.37, and `diversity_index = 0.863` — essentially random. The pheromone table
has not begun to converge on a graph this size within 40 steps, which is the
budget argument in 18.3 made larger and more visible.

### 17.5 Convergence after failure

`routing/failover.py` measures something none of the latency columns can: how
many ticks a router needs to restore a *QoS-satisfying* route after a link on an
active flow is cut.

`measure_convergence()` records the route before the cut, injects the failure,
then steps the simulator until the router produces a path that is both
successful and feasible under the flow's profile. It returns
`convergence_steps` together with `latency_before` and `latency_after`, so a
fast-but-worse recovery is distinguishable from a slow-but-better one — a
distinction a single "convergence time" number destroys.

The `FailoverMonitor` runs the same logic continuously against watched flows and
emits `RerouteEvent`s, which the dashboard's Failover panel renders live. The
important design decision is that it detects breakage by checking
`path_is_intact()` **and** re-evaluating the QoS profile: a path whose links all
still exist but whose bottleneck utilisation has crossed the class's threshold
is broken for that class, even though every link is up.

### 17.6 The result that was not there

One finding in this project is not a measurement. It is a discrepancy.

`docs/12_KNOWN_ISSUES.md` stated that the PPO agent's "mean reward improved from
−77 to −61 (+21%)" with a "best evaluation reward at −45.81". The committed
`runs/ppo_routing/evaluations.npz` contained values from **−86.57 to −99.67**,
with the best result at the **first** checkpoint (25k steps) and the worst at
−99.67. The three headline numbers appear nowhere in the artifact.

Nobody wrote them dishonestly. They came from a training run that was never
committed, and the artifact was later replaced by a different run's output.
There was simply no mechanism that could ever notice.

`scripts/verify_claims.py` is that mechanism, and CI runs it on every push. It
re-reads the artifacts and fails if a documented number is not present in them.
It also checks that every model declared as shipping in `ml/model_registry.py`
actually exists on disk — the check that would have caught
`rl_router_final.zip` being loaded when the file on disk was
`ppo_routing_agent.zip`, a mismatch that meant the RL row in every published
result had been produced by a fallback heuristic.

The lesson generalises past this repository: **a number in a document that no
process can contradict is not a result, it is a memory.**

---

## 18. Discussion

### 18.1 The central negative result

The GNN reproduces Dijkstra's path 96–98% of the time on best-effort traffic,
and its latency is statistically indistinguishable from Dijkstra's. The
benchmark flags this as degeneracy and writes a warning into every results file.

It is worth being very clear that **this is the correct outcome, and it is a
finding rather than a failure.**

Section 2.1 establishes the reason. With edge costs that are non-negative and
additive along a path, Dijkstra returns the minimum-cost path — a theorem, not a
heuristic. Both conditions hold for `link_cost(link) = base_latency × (1 + 4u²)`:
it is positive for all `u`, and a path's cost is the sum of its links'. There is
no path better than Dijkstra's. A learned router that reached 98% agreement did
not fail to beat Dijkstra; it succeeded at finding the optimum, by a different
and much more expensive route.

The project's original framing — "AI-based routing outperforms classical
algorithms" — was therefore unfalsifiable in the regime it was tested in. Not
merely unsupported: *unachievable*. No amount of training, no architecture, no
hyperparameter sweep can produce a path cheaper than the cheapest path.

What the 98% figure actually establishes is that the GNN learned the cost
structure of the network well enough to rediscover the optimal policy from data,
with a top-1 accuracy of 0.978 and a mean regret of 0.00064 on paths it had
never seen. That is a legitimate machine-learning result. It is a statement
about *learning*, not about *routing*, and the two get conflated constantly in
this literature.

The correct response is not to tune the model. It is to change the regime — which
is what QoS-constrained routing (7, 17.4) does, and why it is now the scenario
the project's thesis rests on.

### 18.2 Where learned routing can actually win

Four conditions, each of which breaks one assumption of the optimality argument:

1. **Non-additive or multiple constraints.** Bottleneck utilisation is a `max`,
   not a sum. Jitter and loss budgets are separate additive constraints. Routing
   subject to two or more independent constraints is NP-hard (Wang & Crowcroft,
   1996), so every practical method is a heuristic and a learned heuristic is
   competing on level ground.
2. **State the classical algorithm cannot see.** Dijkstra is optimal *given the
   cost function it is handed*. If tomorrow's congestion is predictable from
   today's pattern, a forecaster changes the costs Dijkstra is optimising over,
   and the combination can beat Dijkstra-on-current-state. The LSTM's +0.1497
   skill score is small, but it is the mechanism by which a learned component can
   legitimately improve a provably optimal algorithm: not by out-searching it, by
   better-informing it.
3. **Amortised computation.** Dijkstra is `O(E log V)` *per query*. A trained
   network is one forward pass. On a 10,000-node topology with tens of thousands
   of path queries per second, "identical quality at a fraction of the latency"
   is a real operational win even with zero quality improvement. This project
   does not measure inference cost, which is a gap flagged in 21.
4. **Objectives that are not sums of link costs.** Minimising the *maximum* link
   utilisation across the network — the standard traffic-engineering objective —
   is not a shortest-path problem at all. It is a global optimisation over
   simultaneous flows, and greedy per-flow shortest-path routing is provably not
   optimal for it. (This was explicitly scoped out of the current work and is
   listed in 21 as the highest-value extension.)

### 18.3 What ACO and multi-agent are telling you

ACO and the multi-agent policies both land 61–62% worse than Dijkstra, and it is
tempting to read that as "these methods do not work". The diagnostics say
something more specific.

**ACO** shows `diversity_index = 0.70` and `mean_hops = 4.24` against Dijkstra's
0.27 and 2.70. It is exploring widely and taking long paths — the signature of a
pheromone table that has not converged within the evaluation horizon. ACO is an
anytime algorithm whose quality is a function of iteration count; 40 steps is
simply not enough for the pheromone distribution to sharpen. The honest
statement is "ACO is not competitive **at this iteration budget**", and the
budget belongs in the sentence.

**Multi-agent** is more interesting, because its component evaluation (17.1) is
excellent — all four regions beat random by 69–81 return points — while its
system-level latency is 61% worse than Dijkstra. Both numbers are correct, and
the gap between them is the finding: **each agent is good at its local task, and
the composition of locally-good decisions is a globally poor path.**

This is exactly the pathology CTDE is designed to mitigate and does not
eliminate. Each regional agent optimises the variance of its own region's
utilisation. No agent optimises end-to-end latency, because no agent can see an
end-to-end path — that is the entire point of decentralised execution. The
hop-by-hop walk (14.3) composes four myopic policies, and the result is 3.67
hops where the optimum is 2.70.

That is not a bug to be fixed with more training. It is the cost of the
constraint, and it is the honest answer to "why not just do everything with
multi-agent RL?": because decentralisation buys scalability and failure
isolation, and it pays for them in path quality. `ml/cards/regional_experts.md`
states this in the model card so it cannot be quietly forgotten.

### 18.4 Threats to validity

**The simulator is not a network.** It has no packets, no queues, no TCP, no
protocol overhead. `base_latency × (1 + 4u²)` is a plausible congestion curve,
not a measured one. Every result in Part IV is a statement about *this model*,
and generalisation to real networks is an assumption, not a finding. This is the
single largest threat and no amount of statistical care reduces it.

**n = 15.** Correct, but small. A true 1% latency improvement would not be
detectable at this sample size, and the CIs are correspondingly wide. Where the
text says "indistinguishable", read "indistinguishable at n = 15", not "equal".

**One topology family.** Watts–Strogatz small-world, degree 4. Real ISP
topologies are closer to scale-free with a distinct core/edge structure;
data-centre fabrics are highly regular Clos networks. Conclusions about path
diversity and about how much room a learned router has to differ from shortest
path are likely sensitive to this choice.

**Trained and tested on the same generator.** The models see topologies drawn
from the same process that generates the evaluation topologies. Seeds differ, so
this is not leakage in the strict sense, but it is not distribution shift either,
and real deployment is nothing but distribution shift.

**The candidate set bounds everything.** All routers except Bellman–Ford choose
from `candidate_paths(...)`, which returns the k best paths under a
congestion-weighted metric. Nothing can select a path outside that set — not the
GNN, not PPO, not the oracle. A "learned" improvement is therefore always a
*re-ranking* of a classically generated set, which meaningfully limits the
ceiling and is the reason `constrained` is so hard to beat.

---

# Part V — The world

---

## 19. Real-world applications

The honest framing first: **this repository is not a product, and nothing in it
should be pointed at a production network.** What it is, is a correct
implementation of a control-plane pattern — measure the network, featurise it,
score candidate paths, choose one, watch what happens — plus a benchmarking
methodology strict enough to tell you whether the learned part is doing
anything. Both of those transfer. The simulator does not.

### 19.1 Where the pattern maps

**SD-WAN path selection.** This is the closest real analogue, and it is not close
by accident: a branch office with a broadband link, an LTE link and an MPLS
circuit is a tiny graph with a handful of paths, per-application QoS classes
(voice, video, bulk sync, guest), and per-path measurements that already exist
because the vendor's appliances probe continuously. The decision is
"which overlay tunnel carries this application right now" — a candidate-set
re-ranking problem with hard constraints, which is precisely the shape of
`select_best_path()`. Commercial products (Cisco vManage, Velocloud, Fortinet
SD-WAN) implement rule-based versions today; the learned version is a research
frontier rather than a solved problem. **This is where a project like this could
plausibly become real.**

**Data-centre traffic engineering.** ECMP hashes flows across equal-cost paths
and is famously bad at elephant flows: two large flows can hash onto the same
link while parallel capacity sits idle. Systems like Hedera and CONGA address
this with centralised or in-fabric rerouting. The objective here is
min-max utilisation, not per-flow latency — the fourth condition in 18.2 — and
the topology is regular (Clos), which changes the problem substantially. The
relevant transferable piece is the closed-loop evaluation methodology, not the
router.

**ISP traffic engineering with segment routing.** A carrier computes label
stacks (SR-TE policies) to steer traffic along specific paths, subject to
latency and bandwidth SLAs. Path computation is centralised in a PCE (Path
Computation Element) and programmed via PCEP. This is genuinely a
multi-constrained routing problem, genuinely NP-hard, and genuinely solved with
heuristics today. It is the best theoretical fit for the QoS argument in 18.2
and the worst practical fit for a student project, because the consequences of a
bad path are measured in customer SLA penalties.

**CDN and anycast steering.** Choosing which edge PoP serves a user, and over
which transit, based on measured RTT and loss. This is a much softer target
than packet routing — decisions are per-session rather than per-packet, the
control loop runs in seconds, and mistakes degrade rather than break. Learned
components are already deployed in production at this layer.

**LEO satellite constellations.** Starlink-class constellations have topologies
that change *deterministically but constantly* as satellites move. Inter-satellite
link availability is a function of orbital mechanics, which means it is
**predictable** — the ideal case for the forecasting component in 18.2 condition
2, because tomorrow's topology genuinely is derivable from today's.

**5G network slicing.** Slices are literally QoS classes with contractual
guarantees (URLLC: 1 ms, 99.999%; eMBB: throughput; mMTC: density). Routing
subject to per-slice constraints is the multi-constrained problem verbatim,
which is why `core/qos.py` models five classes rather than one.

**Tactical, mesh and disaster-response networks.** Links appear and disappear,
there is no stable central controller, and decisions must be local. This is the
one setting where the *multi-agent* design's weaknesses (18.3) are outweighed by
its properties: decentralised execution keeps working when the network
partitions, and a 30% worse path that exists beats an optimal path that requires
a controller you cannot reach.

### 19.2 What transfers and what does not

| Component | Transfers? | Why |
|---|---|---|
| Benchmark methodology (16) | **Yes, entirely** | Unit of replication, paired tests, floors and ceilings, degeneracy probes are domain-independent. |
| `NetworkSource` abstraction | **Yes** | Swapping simulated / recorded / measured behind one interface is exactly how a real controller is structured. |
| Fallback instrumentation | **Yes** | Any production ML system needs "was this a model decision or a fallback?" as a first-class metric. |
| QoS profile model (`core/qos.py`) | **Mostly** | The five classes map onto DSCP markings and 5G slice types with minor renaming. |
| Cost function | **No** | `1 + 4u²` is a modelling choice. A real deployment measures its latency–load curve per link type. |
| Trained checkpoints | **No** | Trained on synthetic topologies from one generator. They would need retraining on the target network's own traces. |
| Simulator dynamics | **No** | AR(1) around a diurnal cycle is a reasonable caricature and nothing more. |

### 19.3 The economics, briefly

The argument for learned routing in production is rarely "better paths". It is
usually one of:

- **Inference cost.** One forward pass versus a Dijkstra run per query, at
  10⁴–10⁵ queries per second.
- **Constraint handling.** A single model handles five QoS classes; the
  rule-based alternative is five hand-tuned configurations that drift apart.
- **Adaptation.** A model retrained weekly on the operator's own traffic tracks
  changes that a hand-tuned weight does not.

The argument against is that a routing bug is a **network-wide outage**, learned
components fail in ways that are hard to bound, and the classical algorithm is
already optimal for the objective most operators actually configure. This is why
section 20 is mostly about the safety envelope rather than the model.

---

## 20. How you would actually deploy this

### 20.1 Where it sits

A learned router belongs in the **control plane**, never the data plane. It
computes paths; it does not touch packets. The data plane executes the paths it
is given at line rate, and it keeps executing the last good path if the control
plane disappears.

```
        ┌──────────────── control plane ─────────────────┐
        │                                                │
telemetry ──> feature builder ──> model ──> path chooser ──> programmer
   ▲            (ml/features)     (routing/    (core/qos    (PCEP / gNMI /
   │                               learned)     select)      OpenFlow / BGP-LS)
   │                                                │        │
   └──────────── measured outcomes ─────────────────┘        ▼
                                                       ┌──── data plane ────┐
                                                       │ switches / routers │
                                                       └────────────────────┘
```

The loop in that diagram is the same loop `service/main.py` runs once per tick,
with the simulator replaced by real telemetry and the in-memory state replaced
by a device programmer.

### 20.2 The data path, concretely

**Telemetry in.** Real link utilisation comes from streaming telemetry
(gNMI/OpenConfig subscriptions, sFlow/IPFIX samples, or SNMP polling on older
gear). Topology comes from BGP-LS or the IGP's link-state database. Latency and
loss come from active probes (TWAMP) or in-band telemetry (INT). The output of
this stage is exactly the `NetworkState` object this repository already defines:
nodes, links, per-link utilisation, latency, loss, jitter.

**Featurisation.** `ml/features.py` builds the graph tensors and observation
vector from a `NetworkState`. This code transfers unchanged; only the source of
the `NetworkState` changes. This is the single strongest argument for the
`NetworkSource` abstraction — it is the seam along which a research prototype
becomes a controller.

**Inference.** One forward pass per path request, on CPU. Budget: the GNN is
47k parameters and the PPO policy is a small MLP; both are sub-millisecond on a
modern core. For comparison, a Dijkstra run on a 10k-node graph is
single-digit milliseconds. This is the amortisation argument in 18.2 condition 3,
and it is the one a network operator will actually care about.

**Programming the path.** Segment-routing label stacks via PCEP for a carrier;
OpenFlow/P4 table writes for a campus SDN fabric; overlay tunnel selection via a
vendor API for SD-WAN. This layer is entirely absent from this repository and is
where most of the real engineering would go.

### 20.3 The safety envelope

This is the part that makes the difference between a demo and a deployment, and
it is deliberately more detailed than the model section.

**Stage 1 — Shadow mode.** The model runs on live telemetry and logs the path it
*would* have chosen. Nothing is programmed. Run for weeks. The metrics to watch
are agreement rate with the incumbent, and — where they disagree — whether the
model's path would have been better under the metric you actually care about.
This project's `dijkstra_match_rate` probe is precisely a shadow-mode metric.

**Stage 2 — Constrained live, on a slice.** Enable the model for one traffic
class on one region, with hard guardrails:

- Never select a path whose predicted cost exceeds the classical path's by more
  than *x*%.
- Never select a path violating the class's QoS constraints (`evaluate_path`
  already enforces this).
- Never select a path longer than *h* hops.
- Rate-limit changes: at most one reroute per flow per *T* seconds. Route
  flapping does more damage than a suboptimal path.

**Stage 3 — Automatic fallback, always.** If the model fails to load, times out,
produces an infeasible path, or the guardrails reject its choice, fall through to
Dijkstra and **record it**. This is `RouteResult.is_fallback` (16.4) promoted to
an operational metric: a fallback rate that climbs is the earliest available
signal that the model has gone stale or the network has drifted outside its
training distribution.

**Stage 4 — A kill switch a human can reach in one command**, that reverts to
pure classical routing, and that is tested regularly rather than assumed to work.

The `GuardrailBadge` component in the dashboard is the pedagogical version of
Stage 2, showing which decisions the guardrails would have rejected.

### 20.4 Operating it

- **Monitor fallback rate, not just latency.** Latency degrades slowly and
  ambiguously; fallback rate spikes.
- **Monitor agreement with the classical baseline.** A model that starts
  disagreeing far more than it used to has either learned something or broken;
  either way a human should look.
- **Retrain on a schedule, evaluate before promoting.** The evaluation gate is
  the same one used here: does the candidate beat the *current production
  policy* on held-out traces, with a paired test and an effect size? Not "did
  training loss go down".
- **Version checkpoints with their feature schema.** The `link_failures` result
  in 17.3 is a live demonstration of what happens when a checkpoint meets an
  observation width it was not trained for. In production that must be a startup
  error, not a silent fallback.
- **Keep the classical path computation running.** It is cheap, it is the
  fallback, and it is the comparison.

### 20.5 Trying it on a real network — what this repo actually supports

The user-facing version of the above, and one of the goals this project was
extended to meet: a way for anyone to point the tool at something real.

**Trace replay (recommended).** `TraceReplaySource` reads JSONL or CSV frames of
per-link measurements and replays them as network states. Record a trace from
your own monitoring — Prometheus, LibreNMS, `ping`/`iperf` logs, anything that
produces per-link latency and utilisation over time — into the documented schema
(`datasets/README.md`), point the dashboard at it, and every algorithm runs
against your measurements. This is the safe path: it involves no probing, no
privileges, and it is reproducible.

**Live probing (opt-in, deliberately limited).** `LiveProbeSource` measures RTT
and loss to a list of hosts you specify, using the unprivileged system `ping`.
It is gated behind `LIVE_PROBE_ENABLED=1` and it is read-only: no scanning, no
host enumeration, no traffic injection, no raw sockets, no root. It only ever
contacts hosts explicitly listed by the operator.

Its honest limitation is documented in `core/sources.py` and repeated here
because it matters: probing *n* hosts from one machine produces a **star**
topology — one centre, *n* leaves, exactly one path per destination. With one
candidate path per pair there is nothing to route. Live mode is therefore a
**visualisation and measurement** feature, showing real latency and loss on a
real graph; it **cannot** benchmark routing algorithms, and the UI says so
rather than displaying a comparison table full of identical rows.

Getting real routing data requires vantage points at multiple network locations,
which is a different project. Saying that plainly is better than shipping a
feature that appears to do something it cannot.

---

## 21. Limitations and future work

### 21.1 Limitations, stated plainly

1. **The simulator is a caricature.** No packets, no queues, no protocols. Every
   quantitative result is conditional on `base_latency × (1 + 4u²)` being a
   reasonable congestion model, which is an assumption and not a measurement.
2. **Fixed-width observations.** The PPO agent's observation is sized for a
   specific node and link count. It cannot transfer across topology sizes, and
   17.3 shows it failing exactly that way under link failures. The GNN is
   structurally size-agnostic and is the right answer here; the RL formulation
   would need a graph-based encoder to match it.
3. **The candidate set is a ceiling.** Every learned router re-ranks a
   classically generated candidate set. None can discover a path outside it.
4. **n = 15 runs.** Small effects are undetectable. This is an honest limit, not
   a hidden one, but it is a limit.
5. **One topology family, one traffic process.** Watts–Strogatz, AR(1) with a
   diurnal cycle. Generalisation beyond that is untested.
6. **Inference cost is never measured.** The amortisation argument in 18.2 is
   asserted, not demonstrated. A wall-clock comparison of a forward pass against
   a Dijkstra run at several graph sizes would be cheap to add and would make
   that argument real.
7. **No multi-flow objective.** Every decision is made for one flow in
   isolation. The standard traffic-engineering objective is global.

### 21.2 Where this should go next, in priority order

**1. Min-max utilisation as a first-class objective.** Replace "minimise this
flow's cost" with "minimise the maximum link utilisation across all
simultaneous flows". This is not a shortest-path problem, greedy per-flow
routing is provably suboptimal for it, and it is the objective real traffic
engineering optimises. It is the single highest-value change in this list
because it is the one that puts a learned method on ground where it can win.
*(Explicitly out of scope for the current work.)*

**2. Measure inference latency.** Cheap, and it converts 18.2's third condition
from an argument into a result.

**3. A graph encoder for the RL agent.** Replace the flat 286-dimensional
observation with the GNN's node embeddings. Fixes limitation 2, makes the agent
size-agnostic, and would let the same checkpoint run on the 25-node and 100-node
scenarios.

**4. Online / continual learning.** Adapt to drift rather than retraining
offline. Requires a safety story that this project does not have — a model that
updates itself in a control plane is a much larger commitment.
*(Explicitly out of scope for the current work.)*

**5. Real traces.** Public topology datasets (Internet Topology Zoo) with
synthetic traffic, or recorded traces from a lab network, would attack
limitation 1 directly.

**6. Multi-path and load splitting.** Real networks split a flow across paths.
Everything here chooses exactly one.

---

## 22. Glossary

**Additive cost** — a path cost that is the sum of its links' costs. The
condition under which Dijkstra is optimal.

**AR(1)** — first-order autoregressive process, `x_t = a·x_{t−1} + (1−a)·μ + ε`.
Models a quantity that drifts smoothly rather than jumping. Here, link
utilisation with `a = 0.85`.

**Bottleneck** — the most-loaded link on a path. A `max` over links, therefore
**not** additive, therefore outside Dijkstra's guarantee.

**Candidate set** — the k paths a router chooses among, generated by k-shortest
paths under a congestion-weighted metric.

**Cliff's delta** — non-parametric effect size in `[−1, 1]`: the probability one
group exceeds the other minus the reverse.

**Closed loop** — routing decisions change future network state. The property
without which congestion-aware routing is untestable.

**CTDE** — Centralised Training, Decentralised Execution. The critic sees global
state during training; each actor sees only local state at inference.

**Degeneracy** — a learned model reproducing a classical algorithm's output, so
it adds no information. Detected here by `dijkstra_match_rate`.

**Diurnal cycle** — the daily rise and fall of network load, modelled as a sine
term in the offered load.

**ECMP** — Equal-Cost Multi-Path. Hashing flows across equal-cost paths.

**Fallback** — a decision produced by the heuristic rather than the model.
Tracked per decision as `is_fallback`.

**MCOP** — Multi-Constrained Optimal Path. Finding a minimum-cost path subject to
two or more independent constraints. NP-hard.

**Message passing** — the GNN operation where each node aggregates its
neighbours' features and updates its own, repeated *k* times to reach *k*-hop
information.

**Normalized score** — `(policy − random) / (oracle − random)`. 0 = no better
than random, 1 = matches the oracle.

**PPO** — Proximal Policy Optimization. On-policy RL algorithm with a clipped
objective bounding how far a policy update can move.

**Pseudo-replication** — treating correlated observations as independent
samples, inflating *n* and shrinking p-values without justification. The
original benchmark's core statistical error.

**QoS class** — a traffic category with its own cost weights and hard
constraints. Five here: real-time, streaming, interactive, bulk, best-effort.

**Regret** — how much worse the chosen path is than the best available one.

**Skill score** — `1 − MSE_model / MSE_baseline`. Positive means the model beats
the baseline; negative means it is worse than doing nothing.

**Small-world** — a graph with high clustering and short average path length.
Generated here with Watts–Strogatz.

**Wilcoxon signed-rank** — non-parametric paired test on the ranks of paired
differences. Used here because *n* is small and normality is unverified.

---

## 23. Further reading

**Routing theory**
- Dijkstra, E. W. (1959). *A note on two problems in connexion with graphs.* The
  original. Two and a half pages.
- Wang, Z. & Crowcroft, J. (1996). *Quality-of-service routing for supporting
  multimedia applications.* The NP-hardness result that section 7 rests on.
- Chen, S. & Nahrstedt, K. (1998). *An overview of quality-of-service routing.*
  The survey to read after Wang & Crowcroft.

**Learned routing**
- Boyan, J. & Littman, M. (1994). *Packet routing in dynamically changing
  networks: a reinforcement learning approach.* Q-routing. Thirty years old and
  still the clearest statement of the idea.
- Valadarsky, A. et al. (2017). *Learning to route.* Notable for being honest
  about when learned routing does **not** help.
- Rusek, K. et al. (2019). *RouteNet: leveraging graph neural networks for
  network modeling and optimization.* The GNN-for-networks reference.
- Almasan, P. et al. (2022). *Deep reinforcement learning meets graph neural
  networks.* The combination this project's GNN and RL components approximate.

**Traffic engineering in practice**
- Al-Fares, M. et al. (2010). *Hedera: dynamic flow scheduling for data center
  networks.*
- Alizadeh, M. et al. (2014). *CONGA: distributed congestion-aware load
  balancing for datacenters.*
- Filsfils, C. et al. *Segment Routing Architecture* (RFC 8402). How paths are
  actually programmed in a modern carrier network.

**Method**
- Hurlbert, S. (1984). *Pseudoreplication and the design of ecological field
  experiments.* Different field, identical mistake; the paper that names the
  error in 16.1.
- Romano, J. et al. (2006). Cliff's delta interpretation thresholds.
- Henderson, P. et al. (2018). *Deep reinforcement learning that matters.* Why
  RL results are so often not reproducible, and what to report instead.
- Lipton, Z. & Steinhardt, J. (2018). *Troubling trends in machine learning
  scholarship.* On the specific ways ML papers overclaim. Read this one last, and
  then reread section 18.1.

---

*End of the learning guide. The companion documents are
`docs/14_RESULTS_AND_FINDINGS.md` (the generated results) and
`docs/15_GOTCHAS.md` (the traps and how to avoid them).*
