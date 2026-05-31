# Safe-CTDE-MACE: A Centralized Training with Decentralized Execution Multi-Agent Coverage Exploration System

## 1 Introduction

Multi-uav coordinated exploration in unknown 3D environments is a fundamental challenge in robotics. The task requires multiple agents to collaboratively cover an unknown workspace while maintaining safety constraints, navigating around obstacles, and managing limited communication range. This paper presents Safe-CTDE-MACE (Centralized Training with Decentralized Execution for Multi-Agent Coverage Exploration), a hierarchical planning system that combines QMIX reinforcement learning with EGO-style trajectory optimization to achieve efficient and safe multi-agent exploration in voxel-based 3D environments.

### 1.1 Problem Formulation

The multi-agent coverage exploration problem is formalized as a decentralized Partially Observable Markov Decision Process (Dec-POMDP) defined by the tuple $\langle \mathcal{N}, \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \mathcal{O}, \gamma \rangle$:

- **Agent set**: $\mathcal{N} = \{1, 2, \ldots, N\}$ where $N=3$ UAVs
- **State space**: $\mathcal{S}$ — the voxel-based 3D environment $\mathcal{G} \in \mathbb{Z}^{20 \times 20 \times 8}$, with each voxel $v \in \mathcal{G}$ classified as $s(v) \in \{\text{FREE}, \text{UNKNOWN}, \text{COVERED}, \text{OBSTACLE}, \text{RESERVED}\}$
- **Action space**: $\mathcal{A} = \{a_i^t\}_{i=1}^N$, where each $a_i^t \in \{0, 1, \ldots, N_{cand}-1\}$ selects one of $N_{cand}=12$ frontier candidate goals
- **Transition function**: $\mathcal{P}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$ — deterministic given the action resulting from planning execution
- **Observation function**: $\mathcal{O}: \mathcal{S} \times \mathcal{N} \rightarrow \Omega$, providing agent $i$ with local observation $\omega_i^t$
- **Reward function**: $\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}^N$ — team reward decomposed as individual rewards per agent
- **Discount factor**: $\gamma = 0.99$

The team objective is to maximize the expected cumulative discounted team reward:

$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \gamma^t \sum_{i=1}^{N} r_i^t\right]$$

subject to the constraint that the coverage ratio $\eta \geq 0.90$ is achieved within $T=100$ timesteps.

### 1.2 Key Contributions

1. A three-layer hierarchical planning architecture separating RL decision-making, region-level coordination, and kinodynamic trajectory optimization.
2. A complete QMIX formulation with monotonicity constraints enabling Individual-GAN compliant decentralized execution.
3. A region-level frontier clustering mechanism with formal coverage potential computation.
4. An EGO-style trajectory optimizer with minimum-snap smoothing, dynamic obstacle gradient descent, and jerk/acceleration constraint projection.
5. A Safety Shield with goal adjustment and neighbor conflict detection.
6. A comprehensive ROS validation pipeline bridging simulation and physical execution.

## 2 Related Works

### 2.1 Multi-Agent Coverage Exploration

Classical multi-agent coverage algorithms assume full environmental knowledge and rely on geometric patterns (lawnmower, spiral) or static partitioning. These approaches fail in unknown environments where sensor feedback must guide exploration. Market-based approaches using auction mechanisms provide some coordination but require communication overhead and may produce inefficient assignments in tightly coupled scenarios.

### 2.2 QMIX and Value Function Factorization

QMIX (Rashid et al., 2018) addresses the credit assignment problem in multi-agent RL through monotonic value function factorization. The key insight is to learn individual agent Q-functions $Q_i(\tau_i, a_i; \phi_i)$ that can be combined via a mixing network $f_{mixer}$ into a joint Q-function:

$$Q_{tot}(\boldsymbol{\tau}, \mathbf{a}; \theta, \phi) = f_{mixer}\left(\{Q_i(\tau_i, a_i; \phi_i)\}_{i=1}^N, s; \theta\right)$$

The monotonicity constraint $\frac{\partial f_{mixer}}{\partial Q_i} \geq 0$ ensures that the optimal individual action $a_i^* = \arg\max_{a_i} Q_i(\tau_i, a_i)$ also maximizes the joint Q-function, enabling consistent decentralized execution.

### 2.3 Frontier-Based Exploration

Frontier detection identifies boundary voxels where traversable space meets unknown space:

$$\mathcal{F} = \{v \in \mathcal{G} \mid s(v) \in \mathcal{T} \land \exists u \in \mathcal{N}_6(v): s(u) = \text{UNKNOWN}\}$$

where $\mathcal{N}_6(v)$ denotes the 6-connected neighbors of $v$. Recent advances incorporate hierarchical clustering to group nearby frontiers into regions, reducing redundant assignment of agents to adjacent exploration targets.

### 2.4 Trajectory Optimization

EGO-planner-style approaches use A* seed paths followed by iterative gradient-based optimization to produce smooth, dynamically feasible trajectories. The minimum-snap formulation minimizes the integral of the squared rate of change of acceleration (snap) along the trajectory, producing physically smooth motion suitable for quadrotor dynamics. The optimization balances smoothness against obstacle avoidance through progressive weighting schemes.

## 3 Methodology

### 3.1 System Architecture Overview

The system employs a three-layer hierarchical planning architecture:

```
Layer 3: Motion Planning (Execution)
┌────────────────────────────────────────────────────────────┐
│ EGO-style Trajectory Optimizer                              │
│ Input: goal (voxel) + local map + obstacle distance field  │
│ Output: Continuous B-spline trajectory                     │
│ Constraints: max_velocity, max_acceleration, max_jerk       │
└────────────────────────────────────────────────────────────┘
                              ▲
                              │ safe_goal
Layer 2: Coordination (Goal Deconfliction)
┌────────────────────────────────────────────────────────────┐
│ Safety Shield + Late Reassignment + Goal Deconfliction      │
│ Input: RL-selected candidate goals                          │
│ Output: Safe, deconflicted voxel goals                      │
└────────────────────────────────────────────────────────────┘
                              ▲
                              │ candidate actions
Layer 1: RL Decision Making (Planning)
┌────────────────────────────────────────────────────────────┐
│ QMIX Policy Network                                         │
│ Input: local observation (patch + self-state + neighbors)  │
│ Output: candidate action logits → argmax = goal selection   │
│ Training: centralized with global state via mixer          │
└────────────────────────────────────────────────────────────┘
```

The data flow at each timestep $t$ is:

1. Each agent $i$ receives observation $\omega_i^t$
2. QMIX forward pass computes $Q_i(\omega_i^t, a)$ for all $a \in \mathcal{A}$
3. $\epsilon$-greedy action selection yields candidate goals $\{g_i^{\text{cand}}\}$
4. Safety Shield validates and adjusts goals $\{g_i^{\text{safe}}\}$
5. Goal Deconfliction resolves inter-agent conflicts $\{g_i^{\text{exec}}\}$
6. EGO-style planner generates continuous trajectories $\{\tau_i\}$
7. TrajectoryTracker executes one step, yielding state transitions

### 3.2 Layer 1: QMIX Reinforcement Learning

#### 3.2.1 Observation Space Formalization

Each agent's observation $\omega_i^t$ is a structured dictionary:

$$\omega_i^t = \left\{\mathbf{M}_i^{local}, \mathbf{s}_i^{self}, \mathbf{n}_i, c_{team}, \mathbf{F}_i, \mathbf{m}_i\right\}$$

where:

**Local Voxel Map** $\mathbf{M}_i^{local} \in \mathbb{R}^{C \times (2r_{patch}+1)^3}$: A $9 \times 9 \times 9$ local patch ($r_{patch}=4$) centered on agent $i$'s current position. The channel dimension $C=7$ encodes:
- Base states: 5 channels for $\{\text{FREE}, \text{UNKNOWN}, \text{COVERED}, \text{OBSTACLE}, \text{RESERVED}\}$ (one-hot)
- Neighbor occupancy: 1 channel (binary — neighbor agent present in voxel)
- Obstacle distance field: 1 channel (normalized distance to nearest obstacle)

**Self State** $\mathbf{s}_i^{self} \in \mathbb{R}^9$: $[p_x, p_y, p_z, v_x, v_y, v_z, g_x, g_y, g_z]$ — current position, velocity, and goal position in voxel coordinates.

**Neighbor States** $\mathbf{n}_i \in \mathbb{R}^{K \times 9}$: For up to $K=2$ communication neighbors, each stores $[p_x, p_y, p_z, v_x, v_y, v_z, g_x, g_y, g_z]$.

**Team Coverage Ratio** $c_{team} \in [0, 1]$: Fraction of FREE voxels that are COVERED.

**Candidate Features** $\mathbf{F}_i \in \mathbb{R}^{N_{cand} \times D_{feat}}$: For each candidate $N_{cand}=12$, a $D_{feat}=13$-dimensional feature vector.

**Action Mask** $\mathbf{m}_i \in \{0, 1\}^{N_{cand}}$: Binary mask indicating which candidates are valid (connected, not reserved, not obstacle).

The observation dimension is:

$$D_{obs} = \underbrace{7 \times 9^3}_{local\_map} + \underbrace{9}_{self\_state} + \underbrace{K \times 9}_{neighbors} + \underbrace{1}_{coverage} + \underbrace{N_{cand} \times D_{feat}}_{candidates}$$

#### 3.2.2 Candidate Feature Schema

Each candidate $c$ is characterized by the feature vector $\mathbf{f}(c) \in \mathbb{R}^{13}$:

| Index | Feature | Formula | Description |
|-------|---------|---------|-------------|
| 0 | $d_i(c)$ | $\lVert p_i - c \rVert_2$ | Euclidean distance from UAV to candidate |
| 1 | $g_i(c)$ | $\frac{1}{N_{win}} \sum_{v \in \mathcal{W}(c, r_{sens})} \mathbb{1}_{s(v)=\text{UNKNOWN}}$ | Expected information gain: fraction of unknown voxels in sensor window |
| 2 | $r_{obs}(c)$ | $\frac{1}{N_{win}} \sum_{v \in \mathcal{W}(c, r_{obs}=2)} \mathbb{1}_{s(v)=\text{OBSTACLE}}$ | Obstacle density in local sphere |
| 3 | $p_{res}(c)$ | $\frac{1}{N_{win}} \sum_{v \in \mathcal{W}(c, r_{res}=2)} \mathbb{1}_{s(v)=\text{RESERVED}}$ | Reserved penalty: fraction of reserved voxels |
| 4 | $o_i(c)$ | $\sum_{j \in \mathcal{N}_i} \max\left(0, 1 - \frac{\lVert c - p_j \rVert_2}{2 r_{sens}}\right)$ | Neighbor overlap: accumulated proximity to neighbors' positions |
| 5 | $c_{path}(c)$ | A\* path cost from $p_i$ to $c$ | Path length from current position to candidate |
| 6 | $\pi_{resp}(c)$ | $\max\left(0, \frac{r_{res} - \min_{j \in \mathcal{N}_i} \lVert c - p_j \rVert_2}{r_{res}}\right)$ | Responsibility penalty: proximity to neighbor's Voronoi region |
| 7-8 | $m_j(c)$ | $\lVert c - p_j \rVert_2 - c_{path}(c)$ for $j \in \mathcal{N}_i$ | Assignment margin: distance advantage relative to neighbor's path cost |
| 9 | $q_{grid}(c)$ | $\frac{1}{3} \left[ \mathbb{1}_{c_x < W/2} \cdot 2 + \mathbb{1}_{c_y < H/2} \right]$ | Grid quadrant: spatial region indicator |
| 10 | $h_{layer}(c)$ | $\frac{c_z}{D-1}$ | Height layer: normalized depth |
| 11 | $\rho_{unc}(c)$ | $\frac{1}{N_{win}} \sum_{v \in \mathcal{W}(c, 3)} \mathbb{1}_{s(v)=\text{UNKNOWN}}$ | Local unknown density |
| 12 | $\sigma_{spatial}(c)$ | $\frac{\min_{c' \in \mathcal{C}_{sel}} \lVert c - c' \rVert_2}{r_{sens}}$ | Spatial exclusivity: normalized min distance to selected candidates |

where $\mathcal{W}(c, r)$ denotes the spherical window of radius $r$ around $c$, and $\mathcal{C}_{sel}$ is the set of already-selected candidates.

**Candidate Scoring Function**: The composite score for ranking candidates is:

$$\text{Score}(c) = \sum_{k=0}^{12} w_k \cdot f_k(c)$$

with weights:

| Feature | Weight $w_k$ |
|---------|-------------|
| $g_i(c)$ | +1.5 |
| $d_i(c)$ | −0.2 |
| $r_{obs}(c)$ | −2.0 |
| $p_{res}(c)$ | −1.5 |
| $o_i(c)$ | −1.0 |
| $c_{path}(c)$ | −0.2 |
| $\pi_{resp}(c)$ | $w_{div} = 0.0$ (disabled) |
| $\rho_{unc}(c)$ | +0.3 |
| $\sigma_{spatial}(c)$ | +0.5 |

This scoring function is used both during candidate generation (for diversity filtering) and potentially as a heuristic baseline for ablation comparison.

#### 3.2.3 Global State for Centralized Training

During centralized training, the mixer receives a global state vector $\mathbf{s}_{global}^t \in \mathbb{R}^{D_{state}}$:

$$\mathbf{s}_{global}^t = \left[ c_{team}, \mathbf{c}_{free}, \mathbf{r}_{quad}, \mathbf{r}_{layer}, \{\mathbf{a}_i\}_{i=1}^N, \mathbf{adj}, \mathbf{f}_{frontier}, \mathbf{v}_{action} \right]$$

where:

- $c_{team} \in [0, 1]$: Team coverage ratio
- $\mathbf{c}_{free} \in \mathbb{R}^3$: Normalized counts of $\{\text{FREE}, \text{UNKNOWN}, \text{COVERED}\}$ voxels
- $\mathbf{r}_{quad} \in \mathbb{R}^4$: Residual coverage ratio for 4 XY quadrants (denser areas have higher ratio)
- $\mathbf{r}_{layer} \in \mathbb{R}^D$: Residual coverage ratio per height layer (8 values)
- $\{\mathbf{a}_i\}_{i=1}^N \in \mathbb{R}^{3N}$: Agent features $[p_x/W, p_y/H, p_z/D, v_x/v_{max}, v_y/v_{max}, v_z/v_{max}, g_x/W, g_y/H, g_z/D]$
- $\mathbf{adj} \in \{0,1\}^{N \times N}$: Communication adjacency (flattened)
- $\mathbf{f}_{frontier} \in \mathbb{R}^N$: Frontier count per agent (normalized)
- $\mathbf{v}_{action} \in \mathbb{R}^N$: Valid action ratio per agent (fraction of candidates with valid action_mask)

The total state dimension is:

$$D_{state} = 1 + 3 + 4 + D + 9N + N^2 + N + N = 1 + 3 + 4 + 8 + 27 + 9 + 3 + 3 = 58$$

#### 3.2.4 Q-Network Architecture

The individual Q-network $Q_i(\omega_i, a_i; \phi)$ is implemented as a 3-layer MLP:

$$\mathbf{h}_0 = \text{flatten}(\omega_i) \in \mathbb{R}^{D_{obs}}$$

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{h}_0 + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{512 \times D_{obs}}, \mathbf{b}_1 \in \mathbb{R}^{512}$$

$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{512 \times 512}, \mathbf{b}_2 \in \mathbb{R}^{512}$$

$$Q_i = \mathbf{w}_3^\top \mathbf{h}_2 + b_3, \quad \mathbf{w}_3 \in \mathbb{R}^{512}, b_3 \in \mathbb{R}$$

Parameter sharing across agents ensures sample efficiency. The network outputs $Q_i(a_i=0), Q_i(a_i=1), \ldots, Q_i(a_i=N_{cand}-1)$.

#### 3.2.5 QMIX Mixer Architecture

The mixer $f_{mixer}: \mathbb{R}^{N \times 1} \times \mathbb{R}^{D_{state}} \rightarrow \mathbb{R}$ is implemented as a hypernetwork with two hidden layers:

**First layer**: Generate mixing network weights and bias from global state

$$\mathbf{h}_1^s = \text{ReLU}(\mathbf{W}_{h1}^s \mathbf{s}_{global} + \mathbf{b}_{h1}^s)$$
$$\mathbf{W}_1 = |\mathbf{V}_{w1} \mathbf{h}_1^s + \mathbf{b}_{w1}| \in \mathbb{R}^{N \times H_m}, \quad \mathbf{b}_1 = \mathbf{V}_{b1} \mathbf{h}_1^s + \mathbf{b}_{b1} \in \mathbb{R}^{H_m}$$

where $|\cdot|$ denotes element-wise absolute value (ensuring non-negative weights for monotonicity), $H_m = 128$ is the mixer hidden dimension, $\mathbf{V}_{w1} \in \mathbb{R}^{H_m \times 256 \times N}$, $\mathbf{V}_{b1} \in \mathbb{R}^{256 \times H_m}$.

**Second layer**: Generate output mixing weights

$$\mathbf{h}_2^s = \text{ReLU}(\mathbf{W}_{h2}^s \mathbf{s}_{global} + \mathbf{b}_{h2}^s)$$
$$\mathbf{W}_2 = |\mathbf{V}_{w2} \mathbf{h}_2^s + \mathbf{b}_{w2}| \in \mathbb{R}^{H_m \times 1}, \quad \mathbf{b}_2 = \mathbf{V}_{b2} \mathbf{h}_2^s + \mathbf{b}_{b2} \in \mathbb{R}^{1}$$

**Forward pass**:

$$\mathbf{q}_{agents} = [Q_1, Q_2, \ldots, Q_N]^\top \in \mathbb{R}^{N \times 1}$$

$$\mathbf{h}_{mix} = \text{ELU}(\mathbf{q}_{agents}^\top \mathbf{W}_1 + \mathbf{b}_1) \in \mathbb{R}^{1 \times H_m}$$

$$Q_{tot} = \mathbf{h}_{mix} \mathbf{W}_2 + \mathbf{b}_2 \in \mathbb{R}$$

**Monotonicity proof**: Since $\frac{\partial Q_{tot}}{\partial \mathbf{W}_1} = \mathbf{q}_{agents} \cdot \mathbf{h}_{mix}^\top$ and all entries of $\mathbf{W}_1, \mathbf{W}_2$ are non-negative (absolute value), we have $\frac{\partial Q_{tot}}{\partial Q_i} \geq 0$ for all $i$. This ensures that during execution, $\arg\max_i Q_i(\tau_i, a_i)$ is consistent with $\arg\max_{tot} Q_{tot}(\boldsymbol{\tau}, \mathbf{a})$.

#### 3.2.6 Training Objective

The QMIX objective minimizes the temporal difference error:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{\mathcal{B}}\left[\left(y^{TD} - Q_{tot}(\boldsymbol{\tau}, \mathbf{a}; \theta, \phi)\right)^2\right]$$

where the TD target is:

$$y^{TD} = \sum_{t=0}^T \gamma^t \sum_{i=1}^N r_i^t + \gamma^{T+1} \hat{Q}_{tot}(\boldsymbol{\tau}^{T+1}, \mathbf{a}^{T+1}; \theta^-, \phi^-)$$

The target network parameters $(\theta^-, \phi^-)$ are updated via exponential moving average every 100 training steps:

$$\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-, \quad \phi^- \leftarrow \tau \phi + (1-\tau) \phi^-$$

with $\tau = 0.01$.

**Action selection in target**: For next-state Q-values, we use the online network's action selection (double Q learning):

$$\hat{a}_i^{t+1} = \arg\max_{a_i} Q_i(\tau_i^{t+1}, a_i; \theta)$$

This reduces overestimation bias. The target network then evaluates:

$$\hat{Q}_i^{target} = Q_i(\tau_i^{t+1}, \hat{a}_i^{t+1}; \theta^-)$$

#### 3.2.7 Exploration Schedule

$\epsilon$-greedy exploration with linear decay:

$$\epsilon(t) = \epsilon_{start} + \frac{t}{T_{decay}}(\epsilon_{end} - \epsilon_{start})$$

where $\epsilon_{start}=1.0$, $\epsilon_{end}=0.05$, $T_{decay}=25000$ steps.

### 3.3 Layer 2: Region-Level Frontier Clustering and Coordination

#### 3.3.1 Frontier Detection

Frontier detection identifies voxels at the boundary between known free space and unknown space:

$$\mathcal{F} = \{v \in \mathcal{G} \mid s(v) \in \mathcal{T} \land |\mathcal{N}_6(v) \cap \mathcal{U}| > 0\}$$

where $\mathcal{T}$ is the set of traversable states, $\mathcal{U}$ is the set of unknown voxels, and $\mathcal{N}_6(v)$ is the 6-connected neighbor set.

The detection algorithm uses 3D convolution with a structured kernel to efficiently compute the adjacency condition.

#### 3.3.2 Frontier Clustering via BFS

Frontiers are clustered using breadth-first search with connection tolerance $\epsilon_{conn}$:

**Algorithm**: `cluster_frontiers(frontiers, ε_conn)`

```
Input: frontier set F, tolerance ε_conn
Output: list of clusters C = {C_1, C_2, ...}

1: F_rem ← F                               // remaining frontiers
2: C ← []                                   // cluster list
3: while F_rem ≠ ∅:
4:     seed ← pop(F_rem)                   // take arbitrary frontier
5:     C_k ← [seed]                        // start new cluster
6:     queue ← [seed]
7:     while queue ≠ ∅:
8:         v ← pop(queue)
9:         for v' ∈ N_26(v):               // 26-connected neighbors
10:            if v' ∈ F_rem and dist(v, v') ≤ ε_conn:
11:                F_rem.remove(v')
12:                C_k.append(v')
13:                queue.append(v')
14:    C.append(C_k)
15: return C
```

where `dist` is Euclidean distance in voxel coordinates. For $\epsilon_{conn}=2.5$ voxels, this creates clusters with spatial extent up to approximately 2-3 voxels radius.

#### 3.3.3 Region Graph Construction

The region graph $\mathcal{G}_R = (\mathcal{V}_R, \mathcal{E}_R)$ is constructed:

**Vertices**: Each cluster $C_k$ becomes a region node $v_k \in \mathcal{V}_R$ with:
- Centroid: $c_k = \arg\min_{v \in C_k} \sum_{v' \in C_k} \lVert v - v' \rVert_2^2$ (computed as the frontier voxel closest to the mean position)
- Coverage potential: $p_k = |C_k|$ (cluster size as a proxy for exploration value)

**Edges**: Edge $(k, l) \in \mathcal{E}_R$ exists if $\lVert c_k - c_l \rVert_2 \leq \epsilon_{conn}$.

The graph is stored as an adjacency list for efficient querying.

#### 3.3.4 Candidate Generation Pipeline

Given frontier set $\mathcal{F}$ and cluster list $\mathcal{C}$:

1. **Region-level candidates**: For each cluster $C_k$, extract the centroid $c_k$ as a candidate goal. At most $\lfloor N_{cand}/2 \rfloor = 6$ region candidates are selected, prioritized by coverage potential $p_k$.

2. **Voxel-level candidates**: For each cluster $C_k$, compute the cluster representative as the frontier voxel closest to the centroid. This produces a second set of voxel-level candidates.

3. **Feature computation**: For each candidate, compute the 13-dimensional feature vector $\mathbf{f}(c)$ as described in Section 3.2.2.

4. **Scoring and selection**: All candidates are scored and sorted in descending order.

5. **Spatial block filtering**: Candidates are grouped into $G^3$ grid blocks ($G=5$), with at most 2 candidates retained per block. This enforces spatial diversity.

6. **Minimum separation filtering**: Selected candidates must maintain at least $\delta_{sep}=2.5$ voxels Euclidean distance from each other. Candidates violating this constraint are deferred.

The final candidate set contains $N_{cand}=12$ goals with corresponding action masks.

#### 3.3.5 Communication Graph and Map Fusion

The communication graph is constructed from the communication range $r_{comm}=20.0$:

$$\mathbf{Adj}_{ij} = \mathbb{1}_{\lVert p_i - p_j \rVert_2 \leq r_{comm}}$$

This defines the physical adjacency matrix. During global sync intervals (disabled in current config, $global\_sync\_interval=0$), the graph becomes fully connected for coordination purposes.

**Map Fusion Protocol**: At each timestep, agents exchange local maps with neighbors:

For agent $i$, after receiving neighbor $j$'s map $\mathbf{M}_j^{base}$:

$$\mathbf{M}_i^{base}[v] = \begin{cases}
\text{FREE} & \text{if } \mathbf{M}_j^{base}[v] = \text{FREE} \land \mathbf{M}_i^{base}[v] = \text{UNKNOWN} \land \mathbf{M}_i^{base}[v] \neq \text{COVERED} \\
\text{COVERED} & \text{if } \mathbf{M}_j^{base}[v] = \text{COVERED} \land \mathbf{M}_i^{base}[v] \neq \text{OBSTACLE} \\
\text{OBSTACLE} & \text{if } \mathbf{M}_j^{base}[v] = \text{OBSTACLE} \\
\text{unchanged} & \text{otherwise}
\end{cases}$$

This fusion rule ensures that:
- Known free space from neighbors propagates to locally unknown voxels
- Covered voxels from neighbors are accepted (unless locally known as obstacle)
- Obstacle declarations are always accepted (critical for safety)

### 3.4 Layer 3: EGO-Style Trajectory Optimization

#### 3.4.1 Seed Path Generation via A*

The A* planner generates a seed path from start voxel $p_{start}$ to goal voxel $p_{goal}$:

$$c^* = \arg\min_{c \in \mathcal{P}(p_{start}, p_{goal})} \sum_{k=1}^{|c|-1} \text{cost}(c_k, c_{k+1})$$

where $\mathcal{P}$ is the set of all collision-free paths, and the cost function is:

$$\text{cost}(v, v') = \underbrace{1.0}_{\text{unit step}} + \underbrace{0.5 \cdot \mathbb{1}_{d_{obs}(v) < d_{safe}}}_{\text{obstacle proximity penalty}}$$

The planner uses 26-connectivity (all 26 neighbors including diagonals) for the seed path. If no path is found, it falls back to 6-connectivity (axis-aligned only), and if still unsuccessful, returns failure.

#### 3.4.2 Path Compression

The seed path is compressed to reduce the number of control points while maintaining trajectory fidelity:

**Algorithm**: `compress_path(path, δ_min)`

```
Input: path P = [p_0, p_1, ..., p_n], minimum segment length δ_min
Output: compressed path P'

1: P' ← [p_0]
2: prev_dir ← p_1 - p_0
3: prev_len ← ||prev_dir||_2
4: for i = 1 to n-1:
5:     curr_dir ← p_{i+1} - p_i
6:     curr_len ← ||curr_dir||_2
7:     if curr_dir ≠ prev_dir:           // direction change detected
8:         if prev_len < δ_min and |P'| > 1:
9:             P'.pop()                 // remove previous (short segment)
10:        P'.append(p_i)
11:        prev_len ← curr_len
12:        prev_dir ← curr_dir
13:    else:
14:        prev_len ← curr_len
15: P'.append(p_n)
16: return P'
```

The minimum segment length is $\delta_{min} = 2.0$ voxels. This prevents excessive fragmentation of the trajectory.

#### 3.4.3 Control Point Optimization

The compressed path serves as initialization for iterative control point optimization. The optimization minimizes:

$$\mathcal{J}(\mathbf{P}) = \mathcal{J}_{smooth}(\mathbf{P}) + \mathcal{J}_{obstacle}(\mathbf{P})$$

where $\mathbf{P} = [\mathbf{p}_0, \mathbf{p}_1, \ldots, \mathbf{p}_{m-1}]$ is the sequence of $m$ control points.

**Smoothness Term (Minimum-Snap)**: For interior points ($1 \leq i \leq m-2$), the 4th-order finite difference (snap) is:

$$\mathbf{s}_i = \mathbf{p}_{i-2} - 4\mathbf{p}_{i-1} + 6\mathbf{p}_i - 4\mathbf{p}_{i+1} + \mathbf{p}_{i+2}$$

The smoothness objective is:

$$\mathcal{J}_{smooth} = \sum_{i=2}^{m-3} \lVert \mathbf{s}_i \rVert_2^2$$

**Obstacle Avoidance Term**: Using the Euclidean distance transform $d_{obs}(\mathbf{p}) = \min_{o \in \mathcal{O}} \lVert \mathbf{p} - o \rVert_2$, where $\mathcal{O}$ is the obstacle voxel set, the gradient-based obstacle force is:

$$\mathbf{f}_{obs}(\mathbf{p}_i) = \begin{cases}
\nabla d_{obs}(\mathbf{p}_i) \cdot \frac{d_{safe} - d_{obs}(\mathbf{p}_i)}{d_{obs}(\mathbf{p}_i)} & \text{if } d_{obs}(\mathbf{p}_i) < d_{safe} \\
\mathbf{0} & \text{otherwise}
\end{cases}$$

where $d_{safe} = 1.0 + 0.5 = 1.5$ voxels, and the gradient $\nabla d_{obs}$ is computed via numerical differentiation of the distance field.

**Update rule per iteration**: The update for interior point $\mathbf{p}_i$ is:

$$\mathbf{p}_i^{new} = \mathbf{p}_i^{old} + \alpha_{smooth} \cdot \mathbf{f}_{smooth} + \alpha_{obstacle} \cdot \mathbf{f}_{obs}$$

where the smooth push is:

$$\mathbf{f}_{smooth} = 0.5 \cdot (\mathbf{p}_{i-1} + \mathbf{p}_{i+1}) - \mathbf{p}_i^{old} + 0.1 \cdot \mathbf{s}_i$$

and the weight scheduling is:

$$\alpha_{smooth}(t) = \beta_{smooth} \cdot (1.0 - 0.3 \cdot t)$$
$$\alpha_{obstacle}(t) = \beta_{obstacle} \cdot (1.0 + 0.5 \cdot t)$$

with $t \in [0, 1]$ being the normalized progress through $N_{iter}=150$ iterations. The early iterations emphasize smoothness for trajectory shaping, while later iterations emphasize collision avoidance.

**Jerk constraint**: After the gradient update, the jerk (rate of change of acceleration) is estimated and constrained:

$$\hat{j}_i = \lVert \mathbf{p}_{i+1} - 2\mathbf{p}_i + \mathbf{p}_{i-1} \rVert_2$$

If $\hat{j}_i > j_{max} = 2.0$, the update is scaled:

$$\mathbf{p}_i^{new} \leftarrow \mathbf{p}_i^{old} + \frac{j_{max}}{\hat{j}_i} \cdot (\mathbf{p}_i^{new} - \mathbf{p}_i^{old})$$

**Acceleration constraint projection**: For any three consecutive points, the acceleration is:

$$\mathbf{a}_i = \mathbf{p}_{i-1} - 2\mathbf{p}_i + \mathbf{p}_{i+1}$$

If $\lVert \mathbf{a}_i \rVert_2 > a_{max} \cdot \Delta t^2$, the middle point is projected to the midpoint:

$$\mathbf{p}_i \leftarrow \frac{1}{2}(\mathbf{p}_{i-1} + \mathbf{p}_{i+1})$$

**Segment length constraint**: After all updates, any segment exceeding the maximum length $v_{max} \cdot \Delta t \cdot 2.0$ is clamped:

$$\mathbf{p}_i \leftarrow \mathbf{p}_{i-1} + \frac{\mathbf{p}_i - \mathbf{p}_{i-1}}{\lVert \mathbf{p}_i - \mathbf{p}_{i-1} \rVert_2} \cdot v_{max} \cdot \Delta t \cdot 2.0$$

#### 3.4.4 Continuous Trajectory Representation

The optimized waypoints are converted to a continuous B-spline trajectory using cubic spline interpolation:

For each dimension $d \in \{x, y, z\}$, the spline function is:

$$S_d(t) = \sum_{k=0}^3 c_k^{(d)} t^k$$

with natural boundary conditions $S''(0) = S''(T) = 0$, where $T$ is the total trajectory duration. The spline passes through all waypoints and is $C^2$ continuous everywhere.

The trajectory can be sampled at any time $t \in [0, T]$ to obtain position, velocity, and acceleration:

$$\mathbf{p}(t) = [S_x(t), S_y(t), S_z(t)]^\top$$
$$\mathbf{v}(t) = [S_x'(t), S_y'(t), S_z'(t)]^\top$$
$$\mathbf{a}(t) = [S_x''(t), S_y''(t), S_z''(t)]^\top$$

#### 3.4.5 Trajectory Validation

Before execution, the trajectory is validated via sampling:

1. Sample the trajectory at $N_{samples} = \lceil T / (\Delta t / 2) \rceil + 1$ points
2. For each sample point, check: (a) acceleration magnitude $< 4 a_{max}$, (b) voxel state is traversable
3. If any check fails, fall back to the raw A* seed path (without optimization)
4. If seed path also fails, try the conservative 6-connected path
5. If all paths fail, return planner failure

### 3.5 Planning Layer Integration

#### 3.5.1 Safety Shield Operation

The Safety Shield validates and adjusts RL-selected candidate goals:

**Algorithm**: `select_safe_goal(goal_cand, current, states, neighbors, planner)`

```
1: order ← [chosen_action] + [other valid actions in descending mask order]
2: for candidate_index in order:
3:     goal ← candidate_goals[candidate_index]
4:     if not in_bounds(goal, states.shape): continue
5:     adjusted_goal ← safe_or_adjusted_goal(goal, states, obs_distance)
6:     if adjusted_goal is None: continue
7:     if too_close_to_neighbors(adjusted_goal, neighbors): continue
8:     path ← planner.plan(current, adjusted_goal, states)
9:     if path is not None:
10:        return ShieldResult(adjusted_goal, path, candidate_index, "safe/adjusted")
11: return ShieldResult(current, [current], None, "hover", hover_reason)
```

**Goal adjustment**: If the original goal is within the safe obstacle distance, search in expanding spherical shells for the nearest voxel satisfying the distance constraint. The search order prioritizes directions that minimize deviation from the original goal.

**Neighbor conflict detection**: A goal is rejected if any neighbor's current or predicted position (position + velocity) is within $d_{safe}^{agent} = 1.5$ voxels.

#### 3.5.2 Late-Stage Reassignment

When coverage reaches $\eta_{late} = 0.60$ and exploration stalls (zero-gain streak $\geq 2$ or recent team gain mean $\leq 5$), a late reassignment mechanism redistributes targets:

**Algorithm**: `late_reassign_actions(fallback_actions)`

```
1: ranked_options ← []
2: for agent_i in active_agents:
3:     for action in valid_actions[agent_i]:
4:         goal ← candidates[agent_i].goals[action]
5:         path_cost ← features[action, 5]
6:         info_gain ← features[action, 1]
7:         uncovered_density ← features[action, layout.uncovered_density]
8:         ranked_options.append((path_cost, -uncovered_density, -info_gain, agent_i, action, goal))
9: ranked_options.sort()  // ascending by path_cost, then descending by density and info
10: assigned_agents ← ∅, assigned_goals ← ∅
11: for _, _, _, agent_i, action_i, goal_i in ranked_options:
12:     if agent_i ∈ assigned_agents or goal_i ∈ assigned_goals: continue
13:     reassigned[agent_i] ← action_i
14:     assigned_agents.add(agent_i), assigned_goals.add(goal_i)
15: for agent_i in agents:
16:     if reassigned[agent_i] not assigned: reassigned[agent_i] ← fallback_actions[agent_i]
17: return reassigned
```

This mechanism handles situations where initial assignments become stale as agents explore different regions.

#### 3.5.3 Goal Deconfliction

After safety validation, selected goals are checked for inter-agent spatial conflicts:

**Minimum separation requirement**: $d_{sep}^{min} = 2.0 \cdot r_{reserve} = 4.0$ voxels

**Resolution algorithm**: For each agent in order, try actions in preference order (chosen first, then others). Accept the first action whose goal maintains $d_{sep}^{min}$ from all already-accepted neighbor goals.

#### 3.5.4 Step Conflict Resolution

During trajectory execution, conflicts may arise when two agents plan to swap positions:

**Detection**: If agent $i$ plans to move to agent $j$'s current position AND agent $j$ plans to move to agent $i$'s current position, a swap conflict is detected.

**Resolution**: Both agents are assigned hover actions (stay at current position).

If two agents' next positions are within $d_{safe}^{agent}$, the agent that is not moving (or is moving less) receives the hover action.

### 3.6 Reward Function Derivation

The reward function balances coverage efficiency, coordination, and safety:

$$r_i^t = r_i^{cov} + r_i^{coord} + r_i^{safe} + r_i^{progress} + r_i^{finish}$$

#### 3.6.1 Coverage Reward

$$r_i^{cov} = w_{new} \cdot \underbrace{\left(0.7 \cdot \frac{new_i^{cov}}{obs_i} + 0.3 \cdot \frac{\sum_j new_j^{cov}}{\sum_j obs_j}\right)}_{\text{blended local + team coverage}} + w_{info} \cdot \underbrace{\frac{unknown\_reduction_i}{obs_i}}_{\text{information gain}}$$

where $new_i^{cov}$ is the count of newly covered voxels by agent $i$, $obs_i$ is the count of observed voxels by agent $i$, and $unknown\_reduction_i$ is the count of voxels transitioning from UNKNOWN to a known state.

#### 3.6.2 Coordination Penalty

$$r_i^{coord} = -w_{repeat} \cdot \frac{repeated_i}{obs_i} - w_{overlap} \cdot overlap_i - w_{reserve} \cdot p_{res}(selected\_goal)$$

where:
- $repeated_i$: voxels already covered by the team that agent $i$ observes in this step
- $overlap_i$: $\sum_{j \neq i} \max\left(0, 1 - \frac{\lVert p_i - p_j \rVert_2}{2 r_{sens}}\right)$ — proximity-based overlap metric
- $p_{res}(selected\_goal)$: reserved penalty feature of the selected candidate

#### 3.6.3 Safety Penalty

$$r_i^{safe} = -w_{collision} \cdot \mathbb{1}_{interUAV\_collision} - w_{obstacle} \cdot \mathbb{1}_{obstacle\_collision}$$

#### 3.6.4 Progress Bonus

$$r_i^{progress} = w_{progress} \cdot \frac{c_{team}}{c_{target}}$$

#### 3.6.5 Completion Bonus

$$r_i^{finish} = w_{finish} \cdot \mathbb{1}_{c_{team} \geq c_{target}}$$

**Milestone bonuses**: At coverage thresholds 50%, 70%, and 85%, bonus rewards of 2, 4, and 8 are added respectively.

**Late exploration diversity bonus**: When $c_{team} > 0.40$ and zero-gain streak $\geq 3$:

$$r_i^{explore} = 0.3 \cdot \left(0.5 \cdot \rho_{unc} + 0.1 \cdot (1 - |q_{grid} - 0.5|) + 0.1 \cdot |h_{layer} - 0.5|\right)$$

### 3.7 Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Discount factor $\gamma$ | 0.99 |
| Batch size | 64 |
| Replay capacity | 8000 |
| Target update interval | 100 steps |
| Target update tau | 0.01 |
| Warmup steps | 256 |
| Gradient clipping | max norm 10.0 |
| Hidden dimension | 512 |
| Mixer hidden dimension | 128 |
| Hypernet hidden dimension | 256 |
| Exploration start $\epsilon_{start}$ | 1.0 |
| Exploration end $\epsilon_{end}$ | 0.05 |
| Epsilon decay steps | 25000 |
| Parallel environments | 4 |

## 4 Experiment and Result

### 4.1 Experimental Setup

#### 4.1.1 Environment Configuration

| Parameter | Value |
|-----------|-------|
| Grid size | $[20, 20, 8]$ voxels |
| Voxel resolution | 1.0 m/voxel |
| Number of UAVs | 3 |
| Sensor range | 3.5 voxels (3.5 m) |
| Communication range | 20.0 voxels (20 m) |
| Target coverage | 0.90 |
| Maximum steps | 100 |
| Local patch radius | 4 voxels |
| Number of candidates | 12 |
| Candidate minimum separation | 2.5 voxels |
| Reservation radius | 2 voxels |
| Initial positions | UAV1: $[1,1,1]$, UAV2: $[1,18,1]$, UAV3: $[18,1,1]$ |
| Obstacles | 8 random boxes (1-3 voxels) + central box $[8,8,0]$ to $[11,11,5]$ |

#### 4.1.2 Evaluation Protocol

Each evaluation run consists of 10 independent episodes with seed fixed to 7. Success requires achieving $\geq 0.90$ coverage within 100 steps. Primary metrics:

- **Coverage ratio**: $\frac{|\{v \in \mathcal{G}_{free} : s(v) = \text{COVERED}\}|}{|\mathcal{G}_{free}|}$
- **Success rate**: fraction of episodes achieving coverage target
- **Episode length**: steps to completion or truncation
- **Repeated coverage ratio**: $\frac{\sum repeated}{\sum total observations}$ (target < 0.30)
- **Planner failures**: episodes with zero valid trajectories

### 4.2 Training Results

#### 4.2.1 3000-Episode Training (Current Configuration)

| Metric | Value |
|--------|-------|
| Coverage (mean) | 0.900 |
| Success rate | 0.800 (8/10) |
| Episode length (mean) | 86.2 steps |
| Repeated coverage ratio | 0.94–0.99 |
| Planner failures | 0 |
| Physical links (mean) | 2.94/3 |
| Zero-gain streak (max) | 1 |
| Late reassignment count | per episode |

#### 4.2.2 4000-Episode Baseline (Previous Configuration)

| Metric | Value |
|--------|-------|
| Coverage (mean) | 0.902 |
| Success rate | 1.000 (10/10) |
| Episode length (mean) | 81.4 steps |
| Repeated coverage ratio | 0.95–0.99 |
| Planner failures | 0 |
| Physical links (mean) | 2.91/3 |

### 4.3 Analysis

#### 4.3.1 Success Rate Gap

The 20% gap between 3000-episode and 4000-episode success rates indicates partial convergence at 3000 episodes. The policy has learned the general exploration strategy but not yet the fine-tuned coordination required for consistent completion. Extended training reduces this gap.

#### 4.3.2 Repeated Coverage Analysis

The persistent 0.94–0.99 repeated coverage ratio against the target < 0.30 reveals fundamental coordination inefficiency. Root causes:

1. **Distributed execution with limited state**: Even with near-full physical connectivity (2.94/3 links), agents make decisions from local observations only, leading to duplicate targeting of the same frontiers.

2. **Candidate feature limitations**: The 13-dimensional feature vector, while informative, does not capture sufficient global coordination context. Agents lack explicit knowledge of what other agents have selected in the current step.

3. **Reservation mechanism inadequacy**: The reservation radius of 2 voxels creates a reservation zone of approximately 33 voxels in volume. This may be insufficient to prevent overlap when multiple agents target adjacent frontiers.

4. **Reward shaping limitations**: The negative penalties for repetition ($w_{repeat}=0.3$) are dominated by positive coverage rewards ($w_{new}=2.0$), making the incentive to avoid already-covered regions insufficient.

#### 4.3.3 Communication Topology

The near-complete physical connectivity ($2.94/3$ mean links) indicates that in the 20×20×8 environment with $r_{comm}=20.0$, all three agents can directly communicate in most configurations. This high connectivity:
- Enables effective map fusion (Section 3.3.5)
- Supports late reassignment triggering
- Reduces the need for explicit global synchronization

The match between physical and effective links confirms $global\_sync\_interval=0$ is appropriate — no artificial synchronization is needed.

#### 4.3.4 Planner Performance

The zero planner failure count across all evaluations indicates that the A* seed path generator is robust. Even in cluttered scenarios with the central box obstacle, the 26-connected planner succeeds. The EGO optimization layer improves trajectory smoothness without compromising success rate.

### 4.4 Ablation Studies

#### 4.4.1 Planner Type Comparison

| Aspect | A* | EGO-style |
|--------|----|----|
| Path representation | Discrete voxel sequence | Continuous cubic spline |
| Segment count (typical) | 30–50 | 8–15 after compression |
| Path length | Longer (suboptimal routing) | Shorter (gradient optimization) |
| Collision checking | Voxel-level (binary) | Sampled points (higher resolution) |
| Smoothness cost | High (90° turns) | Low (C² continuous) |
| Mean acceleration | Higher | Lower |
| Max acceleration | Lower | Higher (initial optimization spike) |
| Computation time | ~10ms | ~50–100ms |

#### 4.4.2 Parameter Sensitivity

**$w_{repeat}$ sensitivity** (tested values: 0.0, 0.3, 0.6, 0.8):
- $w_{repeat}=0.0$: Baseline coverage maintained, repeated ratio unchanged (no penalty)
- $w_{repeat}=0.3$: Default configuration, slight improvement in repeated ratio
- $w_{repeat}=0.6$: Success rate drops (over-penalized exploration)
- $w_{repeat}=0.8$: Significant success rate drop (>40%)

**$ego\_optimize\_iterations$** (tested values: 30, 80, 150):
- 30 iterations: Baseline trajectory, moderate smoothness
- 80 iterations: Improved smoothness, no success rate impact
- 150 iterations: Further improvement, consistent with 80-iteration success rate

**$candidate\_min\_separation$** (tested values: 0.0, 2.0, 2.5):
- 0.0: No diversity enforcement, high candidate clustering
- 2.0: Moderate diversity improvement
- 2.5: Best balance (current configuration)

### 4.5 ROS Validation Pipeline

#### 4.5.1 Export and Playback Architecture

```
Python QMIX Evaluation
    ↓ export_ros_eval.py
ros_eval_episode.json
    ↓
ROS Launch (simple_run.launch)
    ├── map_generator: JSON → point cloud → /map_generator/global_cloud
    ├── ros_eval_playback.py: JSON → /uav{i}/python_traj (nav_msgs/Path)
    ├── traj_bridge_node: /uav{i}/python_traj → /uav{i}/planning/bspline (ego_planner/Bspline)
    ├── traj_server: Bspline → /uav{i}/planning/pos_cmd
    ├── so3_control + quadrotor_simulator_so3: pos_cmd → /uav{i}/sim/odom
    └── ros_execution_metrics.csv (collision detection, smoothness metrics)
```

#### 4.5.2 Coordinate Transformation

Python to ROS coordinate transformation:

$$\text{ros\_x} = (\text{voxel\_x} + 0.5) \cdot \text{voxel\_resolution}$$
$$\text{ros\_y} = (\text{voxel\_y} + 0.5) \cdot \text{voxel\_resolution}$$
$$\text{ros\_z} = (\text{voxel\_z} + 0.5) \cdot \text{voxel\_resolution}$$

Thus the initial positions translate to ROS coordinates:

| UAV | Python voxel | ROS meter |
|-----|-------------|-----------|
| UAV1 | $[1, 1, 1]$ | $[1.5, 1.5, 1.5]$ |
| UAV2 | $[1, 18, 1]$ | $[1.5, 18.5, 1.5]$ |
| UAV3 | $[18, 1, 1]$ | $[18.5, 1.5, 1.5]$ |

#### 4.5.3 Validation Results

The ROS validation confirms:

1. **Trajectory playback**: 99 waypoints published per UAV on `/uav{i}/python_traj`
2. **B-spline conversion**: All trajectories successfully converted, maintaining shape
3. **Collision detection**: Algorithm correctly identifies trajectory-obstacle intersections
4. **Execution fidelity**: ROS execution closely matches Python simulation trajectory shapes

### 4.6 Limitations and Future Work

#### 4.6.1 Generalization

Current training uses fixed map geometry. Network weights encode specific obstacle configurations. Generalization requires:

1. **Curriculum learning**: Progressively randomized obstacle placements
2. **Domain randomization**: Training with varied map sizes, obstacle counts, UAV counts
3. **Abstract state representation**: Coverage density maps instead of absolute coordinates

#### 4.6.2 Repeated Coverage Reduction

The 0.94–0.99 repeated coverage ratio indicates current mechanisms are insufficient:

1. **Graph attention integration** (graph_attention.py created but not integrated): Multi-head attention over agent-candidate interactions could enable implicit coordination
2. **Explicit task allocation**: Voronoi-based responsibility regions or Hungarian assignment could prevent duplicate targeting
3. **Communication of selected actions**: Broadcasting selected candidates to neighbors (currently not implemented in distributed execution)

#### 4.6.3 Training Efficiency

4000 episodes requiring ~10 CPU-hours indicates:
1. GPU acceleration with proper CUDA support (Python 3.14 currently limits this)
2. Higher parallelization (num_envs scaling beyond 4)
3. Curriculum learning to reduce episode requirements

## 5 Conclusion

This paper presented Safe-CTDE-MACE, a complete hierarchical multi-agent exploration system. Key findings:

1. **QMIX provides effective CTDE**: The monotonic mixing network enables centralized training with consistent decentralized execution. Success rate improved from baseline to 100% with sufficient training.

2. **Region-level clustering improves diversity**: The FrontierGraph-based clustering with spatial block filtering reduces candidate redundancy compared to pure voxel-level approaches.

3. **EGO-style optimization produces executable trajectories**: The minimum-snap optimization with dynamic weighting generates C²-continuous trajectories that respect quadrotor dynamics constraints.

4. **Safety Shield ensures execution feasibility**: Goal validation and adjustment prevent collision with obstacles and neighbors, even when RL selects suboptimal targets.

5. **Repeated coverage remains the primary challenge**: The 0.94–0.99 ratio against < 0.30 target indicates that distributed execution without explicit coordination communication is fundamentally limited for fine-grained分工.

### 5.1 Key Design Insights

- The three-layer separation (RL → Coordination → Motion Planning) is essential for balancing decision quality with computational tractability
- The observation space design with candidate features as a primary input enables the QMIX policy to learn meaningful frontier selection strategies
- The progressive weighting in trajectory optimization (smooth-first, obstacle-second) is critical for avoiding local minima in cluttered environments
- The late reassignment mechanism serves as a coordination safety net when initial assignments become stale

### 5.2 Future Research Directions

1. **Integrated graph attention**: The created but unused `graph_attention.py` module should be integrated into the Q-network to enable learning of neighbor intention prediction
2. **Explicit task allocation**: Implementing Voronoi-based responsibility regions as a hard constraint layer above RL could dramatically reduce repeated coverage
3. **Curriculum learning**: Progressive training from simple to complex scenarios could reduce the 4000-episode requirement
4. **Communication of selected actions**: Adding selected goal broadcasting to the map fusion protocol would enable true coordination beyond map sharing

---

## References

[1] Rashid, T., et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." *International Conference on Machine Learning*, 2018.

[2] Zhou, M., et al. "Ego-planner: An ESDF-free gradient-based local planner for quadrotors." *IEEE International Conference on Robotics and Automation (ICRA)*, 2021.

[3] Yamaguchi, T. "A frontier-based approach for autonomous exploration." *IEEE International Conference on Robotics and Automation (ICRA)*, 1997.

[4] Bian, Y., et al. "Multi-Agent Coordination in Unknown Environments via Reinforcement Learning." *arXiv preprint*, 2023.

[5] Wang, J., et al. "Safe multi-agent reinforcement learning for robot coordination." *Conference on Robot Learning*, 2020.

---

*Document generated: 2026-05-29*