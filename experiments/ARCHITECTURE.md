# Architecture Design — PETS + EBM Danger-Score Guided Search

## 1. Project Goal Recap

**Goal 1**: Prove that `danger_score = EBM(F1…F6)` can independently guide
a genetic search to find failure-inducing episodes faster than random testing.

- Algorithm: PETS (Probabilistic Ensemble Trajectory Sampling)
- Environment: gymnasium `CartPole-v1` (discrete action space, obs dim=4)
- Fitness: **only** `danger_score` — no reward, no fault probability, no certainty
- No perturbation channels (E1–E4)
- Search framework: adapted from STARLA's multi-objective genetic algorithm

---

## 2. End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 0  Seed Data                          │
│                                                                    │
│   PETS Agent  ──train on CartPole──►  Trained Ensemble + Planner   │
│   Random rollouts ────────────────►  Episode Pool (N episodes)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 1  Feature + Scorer                      │
│                                                                    │
│   Episode Pool ──extract──► Feature Matrix [N × 6]                 │
│   Label episodes ──────────► y ∈ {0=safe, 1=fault}                 │
│   Train EBM  (interpret)  ──► danger_score = EBM.predict_proba(F)  │
│   Train CORELS (corels)   ──► interpretable rule list (optional)   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 2  Guided Search (G1)                    │
│                                                                    │
│   STARLA Genetic Search (adapted)                                  │
│     • Population of Candidates (episodes encoded as init-states)   │
│     • Fitness = danger_score  (single objective, maximize)         │
│     • Selection → Crossover → Mutation → Re-execute via PETS       │
│     • Archive ← episodes where agent actually fails                │
│                                                                    │
│   Random Search Baseline (G0)                                      │
│     • Same budget, random initial states, run PETS                 │
│     • Count faults found                                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 3  Evaluation                            │
│                                                                    │
│   Compare G0 vs G1:                                                │
│     • Time-to-first-fault (episodes until first failure)           │
│     • Total faults found within fixed budget                       │
│     • Fault diversity (unique failure modes)                       │
│   Output: figures + tables → experiments/outputs/                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Responsibilities & File Map

```
experiments/
├── configs/
│   └── default.py              # All hyperparameters (EasyDict)
├── agents/
│   ├── ensemble_model.py       # Ensemble dynamics model (wraps DI-engine EnsembleModel)
│   ├── cem_planner.py          # CEM planner for action selection
│   └── pets_agent.py           # PETS main loop: train world model + plan
├── features/
│   └── feature_extractor.py    # Extract F1–F6 from episode trajectories
├── models/
│   ├── ebm_scorer.py           # Train EBM, expose predict(F) → danger_score
│   └── corels_rules.py         # Train CORELS rule list (optional interpretability)
├── search/
│   ├── candidate.py            # Candidate data structure
│   ├── genetic_search.py       # Main GA loop (adapted from STARLA)
│   ├── operators.py            # Selection, crossover, mutation
│   └── archive.py              # Fault archive management
├── evaluation/
│   ├── baseline_random.py      # G0 random search
│   ├── compare.py              # G0 vs G1 statistics
│   └── plot.py                 # Generate figures
├── data/                       # Runtime artifacts (episodes, models, logs)
├── outputs/                    # Final figures + tables
└── utils/
    ├── env_wrapper.py          # Thin wrapper around gymnasium CartPole
    ├── episode.py              # Episode data structure & serialization
    └── seeding.py              # Reproducibility helpers
```

---

## 4. PETS Implementation Plan

### 4.1 What STARLA Source Tells Us (reference only)

STARLA's code lives in a single notebook: `STARLA/Cart-Pole/STARLA.ipynb`.
It uses a pre-trained Stable-Baselines DQN agent — it does **not** train its own agent.
Key interfaces we observed:

| STARLA component | API | Our equivalent |
|---|---|---|
| `model.predict(obs, deterministic=True)` → (action, _) | Agent policy | `PETSAgent.act(obs)` → action |
| `model.action_probability(state)` → prob[] | Confidence metric | Not needed (no confidence fitness) |
| `model.step_model.step(state)` → Q-values | Abstract-state binning | Replaced (see §6.2) |
| `StoreWrapper(env)` | Trajectory recording | `EpisodeRecorder` wrapper |
| `env.set_state(saved_state)` | State restore for re-exec | `env_wrapper.set_state()` |

### 4.2 Components from DI-engine (reuse)

| Component | DI-engine location | How we use it |
|---|---|---|
| `EnsembleModel` | `ding/world_model/model/ensemble.py` | Core dynamics network — N models predicting (Δobs, reward) with learned variance |
| `StandardScaler` | same file | Input normalization for ensemble |
| `EnsembleFC` | same file | Ensemble-parallel FC layer used internally by EnsembleModel |
| MBPO training loop | `ding/world_model/mbpo.py` `_train()` | Reference for our ensemble training: holdout split, early stopping, elite selection |
| `NaiveReplayBuffer` | `ding/worker/replay_buffer/naive_buffer.py` | Store (obs, act, rew, next_obs, done) transitions |
| `DingEnvWrapper` | `ding/envs/env/ding_env_wrapper.py` | Optional — provides `BaseEnvTimestep` interface |
| `EasyDict` config | `easydict` (external) | Configuration pattern throughout |
| `build_logger` | `ding/utils/log_helper.py` | TensorBoard + text logging |

### 4.3 Components we implement ourselves

| Component | Why not from DI-engine | Complexity |
|---|---|---|
| **CEM Planner** | DI-engine has no CEM/MPC planner at all | Medium — core of PETS |
| **PETS training loop** | DI-engine's `serial_pipeline_dyna` is MBPO-specific (model generates imagined data for SAC). PETS is fundamentally different: pure planning, no policy network | Medium |
| **Episode recorder** | STARLA's `StoreWrapper` depends on gym 0.25 API; we need gymnasium + state save/restore | Low |
| **State save/restore** | CartPole-v1 specific: save/restore `env.unwrapped.state` | Low |

### 4.4 PETS Architecture Detail

PETS differs from MBPO fundamentally:
- **MBPO**: ensemble generates imagined data → trains a model-free policy (SAC)
- **PETS**: ensemble + CEM planner → directly selects actions at each step, **no policy network**

```
                  ┌──────────────────────────────────┐
                  │         EnsembleDynamics          │
                  │  N models, each: (s,a) → (Δs, r) │
                  │  with learned variance            │
                  └──────────┬───────────────────────┘
                             │ predict(obs, action_seqs)
                             │  → mean trajectories + variance
                             ▼
                  ┌──────────────────────────────────┐
                  │          CEM Planner              │
                  │  1. Sample K action sequences     │
                  │  2. Propagate through ensemble    │
                  │  3. Score = mean(sum(rewards))    │
                  │  4. Refit Gaussian to top-J       │
                  │  5. Repeat M iterations           │
                  │  6. Return first action           │
                  └──────────────────────────────────┘
```

**Discrete-action adaptation** (CartPole has 2 actions):

CEM is designed for continuous actions. For CartPole's discrete space, we use one
of these strategies (to be decided during implementation):

- **Option A — Continuous relaxation**: CEM outputs continuous logits ∈ ℝ²,
  Gumbel-softmax samples discrete actions during rollout.
- **Option B — Random shooting**: Skip CEM; sample K random action sequences
  (each action ∈ {0, 1}), evaluate all through ensemble, pick the best.
  Simpler and often sufficient for low-dimensional discrete spaces.
- **Option C — Enumeration-CEM hybrid**: For horizon H and 2 actions,
  the full tree is 2^H. If H ≤ 15, enumerate or subsample heavily.

**Recommendation**: Start with **Option B (random shooting)** for simplicity.
CartPole has only 2 actions and short horizons. If needed, upgrade to Option A.

### 4.5 PETS Training Loop (pseudo-code)

```
1. Initialize ensemble dynamics model (N=5 members)
2. Collect D₀ random transitions from CartPole
3. For epoch = 1 to T:
   a. Train ensemble on replay buffer (early-stop on holdout loss)
   b. For episode = 1 to E:
      - obs = env.reset()
      - For step = 1 to max_steps:
          action = CEM_plan(obs, ensemble, horizon=H)
          next_obs, reward, done, _ = env.step(action)
          buffer.push((obs, action, reward, next_obs, done))
          obs = next_obs
          if done: break
      - Record episode trajectory
4. Return trained ensemble + planner
```

---

## 5. Feature Extraction (F1–F6)

Each episode produces a trajectory: `[(obs₀, a₀, r₀), (obs₁, a₁, r₁), ..., terminal]`.

Candidate features (to be finalized):

| ID | Name | Description | Computation |
|----|------|-------------|-------------|
| F1 | `episode_length` | Steps before termination | `len(trajectory)` |
| F2 | `max_pole_angle` | Max absolute pole angle seen | `max(|obs[2]|)` over trajectory |
| F3 | `max_cart_position` | Max absolute cart displacement | `max(|obs[0]|)` over trajectory |
| F4 | `ensemble_disagreement` | Mean prediction variance across ensemble | Mean of `var(ensemble_preds)` over steps |
| F5 | `action_entropy` | How varied the planned actions are | Entropy of action distribution |
| F6 | `terminal_velocity` | Angular velocity at termination | `|obs[-1][3]|` (last pole angular vel) |

These features capture:
- **Task performance** (F1): short episodes → likely fault
- **State extremity** (F2, F3): large angles/positions → near boundary
- **Model uncertainty** (F4): high disagreement → poor model region
- **Decision difficulty** (F5): high entropy → ambiguous situations
- **Terminal dynamics** (F6): high velocity at end → abrupt failure

---

## 6. STARLA Genetic Search Adaptation

### 6.1 What We Keep from STARLA

| Component | STARLA original | Status |
|---|---|---|
| `Candidate` class | Stores episode + objectives + start_state | **Keep**, simplify |
| `tournament_selection()` | Select best from random subset | **Keep** |
| `generate_offspring_improved_v2()` | 75% crossover + 30% mutation | **Keep** ratios |
| `mutation_improved()` | Perturb state, re-execute from point | **Keep**, extend |
| `re_execute()` | Replay episode from start, continue after mutation | **Keep**, adapt for PETS |
| Archive mechanism | Store fault-triggering episodes | **Keep**, simplify threshold |
| Generation loop | 10 generations | **Keep**, make configurable |

### 6.2 What We Change

#### 6.2.1 Fitness: 3 objectives → 1 danger_score

STARLA uses multi-objective optimization with 3 fitness functions:
```python
# STARLA original (3 objectives, all minimized)
obj_values = [
    fitness_reward(episode),                        # episode length
    fitness_confidence(episode, model, 'm'),         # action uncertainty
    fitness_functional_probability(ml, binary_ep),   # RF fault probability
]
threshold_criteria = [70, 0.04, 0.05]
```

We replace with a **single scalar** (maximized):
```python
# Our adaptation (1 objective, maximized)
features = feature_extractor.extract(episode)   # → [F1..F6]
danger_score = ebm_scorer.predict(features)      # → float ∈ [0, 1]
```

**Consequence**: Multi-objective machinery (preference sort, Pareto front,
crowding distance) becomes unnecessary. We switch to **single-objective
tournament selection** with elitism.

#### 6.2.2 Crossover: abstract-state matching → state-similarity matching

STARLA crossover finds matching crossover points by comparing abstract states
derived from DQN Q-value bins:
```python
# STARLA: requires model.step_model.step() → Q-values
abs_class = abstract_state(model, state, d)   # (ceil(Q1/d), ceil(Q2/d))
```

PETS has no Q-values. We replace with **observation-space proximity**:
```python
# Our adaptation: match by Euclidean distance in obs space
def find_crossover_match(state, other_episode, threshold=0.5):
    for i, (s, a) in enumerate(other_episode):
        if np.linalg.norm(state - s) < threshold:
            return i
    return None  # fallback to random point
```

#### 6.2.3 Mutation: position-only → any state dimension

STARLA only perturbs `state[0]` (cart position) with ±5% noise:
```python
# STARLA
noise = np.random.uniform(0.95, 1.05)
new_state[0] = state[0] * noise
```

We perturb **any dimension** of the initial state (or state at mutation point):
```python
# Our adaptation
dim = np.random.randint(0, obs_dim)          # random dimension
noise = np.random.uniform(0.95, 1.05)
new_state[dim] = state[dim] * noise
```

#### 6.2.4 Episode generation: DQN predict → PETS plan

STARLA re-executes episodes using a trained DQN:
```python
action, _ = model.predict(obs, deterministic=True)
```

We use the PETS agent (CEM/random-shooting planner + ensemble):
```python
action = pets_agent.act(obs)  # CEM plan or random-shooting
```

#### 6.2.5 Archive: objective thresholds → fault detection

STARLA archives candidates that satisfy objective thresholds.
We archive candidates where the **agent actually fails** (episode terminates
before max steps due to pole falling or cart out of bounds):
```python
def is_fault(episode) -> bool:
    return episode.length < MAX_STEPS  # terminated early = fault
```

#### 6.2.6 Selection: preference sort → truncation selection

Since we have a single objective, replace NSGA-II-style preference sort with:
```python
# Sort by danger_score descending, keep top-K
population.sort(key=lambda c: c.danger_score, reverse=True)
next_gen = population[:pop_size]
```

### 6.3 Removed STARLA Components

| Removed | Reason |
|---|---|
| `fitness_confidence()` | Requires `model.action_probability()` — PETS has no policy net |
| `fitness_functional_probability()` | Requires pre-trained RF on binary episode encoding |
| `abstract_state()` + Q-value binning | Requires DQN Q-values — PETS has no value function |
| `translator()` (binary encoding) | Tied to abstract-state vocabulary |
| `preference_sort()` | Multi-objective; we use single-objective |
| `fast_dominating_sort()` | Multi-objective Pareto machinery |
| `crowding_distance` | Multi-objective diversity |

---

## 7. Key Interface Definitions (Pseudo-code)

### 7.1 PETS Agent

```python
# experiments/agents/pets_agent.py

class PETSAgent:
    """Top-level PETS controller: owns ensemble + planner."""

    def __init__(self, config: EasyDict):
        self.ensemble = EnsembleDynamics(config.ensemble)
        self.planner = CEMPlanner(config.planner)  # or RandomShootingPlanner
        self.buffer = NaiveReplayBuffer(config.buffer)

    def train(self, env: gym.Env, num_epochs: int) -> None:
        """Train ensemble on real transitions, collect data with planner."""
        ...

    def act(self, obs: np.ndarray) -> int:
        """Plan next action using ensemble + planner."""
        return self.planner.plan(obs, self.ensemble)

    def rollout(self, env: gym.Env, start_state: Optional[np.ndarray] = None
                ) -> 'Episode':
        """Execute one full episode, return trajectory."""
        ...
```

### 7.2 Ensemble Dynamics Model

```python
# experiments/agents/ensemble_model.py

class EnsembleDynamics:
    """Wraps DI-engine's EnsembleModel with training + prediction interface."""

    def __init__(self, config: EasyDict):
        # Uses ding.world_model.model.ensemble.EnsembleModel internally
        self.model: EnsembleModel  # from DI-engine
        self.scaler: StandardScaler
        self.elite_indices: List[int]

    def train(self, buffer: NaiveReplayBuffer) -> dict:
        """Train ensemble on buffer data. Returns training metrics.
        Reference: MBPOWorldModel._train() in ding/world_model/mbpo.py"""
        ...

    def predict(self, obs: Tensor, action: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict next_obs, reward, done for all ensemble members.
        Returns:
            means:  [ensemble_size, batch, obs_dim+1]
            vars:   [ensemble_size, batch, obs_dim+1]
        """
        ...

    def predict_trajectory(self, obs: Tensor, action_sequences: Tensor,
                           horizon: int) -> Tuple[Tensor, Tensor]:
        """Roll out action sequences through ensemble.
        Args:
            obs:              [batch, obs_dim]
            action_sequences: [batch, horizon, act_dim]
        Returns:
            total_rewards:    [batch]
            disagreements:    [batch]  (mean ensemble variance)
        """
        ...
```

### 7.3 CEM Planner / Random-Shooting Planner

```python
# experiments/agents/cem_planner.py

class RandomShootingPlanner:
    """Sample random action sequences, evaluate via ensemble, pick best."""

    def __init__(self, config: EasyDict):
        self.horizon: int       # planning horizon (e.g., 15)
        self.num_candidates: int  # K action sequences to sample (e.g., 500)
        self.num_actions: int   # 2 for CartPole

    def plan(self, obs: np.ndarray, dynamics: EnsembleDynamics) -> int:
        """Return best first action for given observation."""
        ...


class CEMPlanner:
    """Cross-Entropy Method planner (continuous relaxation)."""

    def __init__(self, config: EasyDict):
        self.horizon: int
        self.num_candidates: int
        self.num_elites: int
        self.num_iterations: int

    def plan(self, obs: np.ndarray, dynamics: EnsembleDynamics) -> int:
        """CEM optimization loop → return first action."""
        ...
```

### 7.4 Feature Extractor

```python
# experiments/features/feature_extractor.py

class FeatureExtractor:
    """Extract 6 danger-relevant features from an episode trajectory."""

    def extract(self, episode: 'Episode',
                dynamics: Optional[EnsembleDynamics] = None
                ) -> np.ndarray:
        """
        Args:
            episode: recorded trajectory
            dynamics: ensemble model (needed for F4 disagreement)
        Returns:
            features: np.ndarray of shape [6]
                [F1_length, F2_max_angle, F3_max_position,
                 F4_disagreement, F5_action_entropy, F6_terminal_velocity]
        """
        ...

    def extract_batch(self, episodes: List['Episode'],
                      dynamics: Optional[EnsembleDynamics] = None
                      ) -> np.ndarray:
        """Extract features for multiple episodes. Returns [N, 6]."""
        ...
```

### 7.5 EBM Danger Scorer

```python
# experiments/models/ebm_scorer.py

class DangerScorer:
    """EBM-based danger score predictor."""

    def __init__(self):
        self.ebm: ExplainableBoostingClassifier  # from interpret

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train EBM on feature matrix [N, 6] and binary labels [N]."""
        self.ebm.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return danger_score (probability of fault class).
        Args:
            features: [N, 6] or [6]
        Returns:
            scores: [N] float in [0, 1]
        """
        return self.ebm.predict_proba(features)[:, 1]

    def explain(self) -> dict:
        """Return EBM feature importances and shape functions."""
        ...
```

### 7.6 Genetic Search (adapted STARLA)

```python
# experiments/search/candidate.py

@dataclass
class Candidate:
    """One individual in the genetic population."""
    episode: 'Episode'          # full trajectory
    start_state: np.ndarray     # initial env state
    danger_score: float = 0.0   # fitness value
    is_fault: bool = False      # did agent actually fail?
    features: Optional[np.ndarray] = None  # [6] cached features


# experiments/search/genetic_search.py

class GeneticSearch:
    """Single-objective GA guided by danger_score."""

    def __init__(self, config: EasyDict,
                 agent: PETSAgent,
                 scorer: DangerScorer,
                 feature_extractor: FeatureExtractor):
        self.pop_size: int          # e.g., 200 (smaller than STARLA's 1500)
        self.num_generations: int   # e.g., 20
        self.crossover_prob: float  # 0.75
        self.mutation_prob: float   # 0.30
        self.tournament_size: int   # 10

    def search(self, env: gym.Env) -> SearchResult:
        """Main search loop.
        Returns:
            SearchResult with archive (fault episodes),
            generation history, total evaluations.
        """
        population = self._init_population(env)
        self._evaluate(population)
        archive = []

        for gen in range(self.num_generations):
            offspring = self._generate_offspring(population, env)
            self._evaluate(offspring)
            population = self._select(population + offspring)
            archive.extend([c for c in offspring if c.is_fault])

        return SearchResult(archive=archive, generations=...)

    def _evaluate(self, candidates: List[Candidate]) -> None:
        """Run episodes, extract features, compute danger_score."""
        for c in candidates:
            c.episode = self.agent.rollout(env, start_state=c.start_state)
            c.features = self.feature_extractor.extract(c.episode)
            c.danger_score = self.scorer.predict(c.features.reshape(1, -1))[0]
            c.is_fault = c.episode.length < MAX_STEPS


# experiments/search/operators.py

def tournament_selection(population: List[Candidate],
                         tournament_size: int) -> Candidate:
    """Select best (highest danger_score) from random subset."""
    ...

def crossover(parent1: Candidate, parent2: Candidate,
              agent: PETSAgent, env: gym.Env) -> Tuple[Candidate, Candidate]:
    """State-similarity crossover: find matching obs, splice, re-execute."""
    ...

def mutate(parent: Candidate, agent: PETSAgent, env: gym.Env) -> Candidate:
    """Perturb random state dimension at random point, re-execute."""
    ...
```

### 7.7 Episode Data Structure

```python
# experiments/utils/episode.py

@dataclass
class Episode:
    """Recorded trajectory from one environment run."""
    observations: np.ndarray    # [T+1, obs_dim]
    actions: np.ndarray         # [T]
    rewards: np.ndarray         # [T]
    total_reward: float
    length: int                 # T (steps before done)
    done: bool                  # terminated by env (not truncated)
    start_state: np.ndarray     # initial env internal state
```

### 7.8 Evaluation / Comparison

```python
# experiments/evaluation/compare.py

@dataclass
class SearchResult:
    archive: List[Candidate]         # fault-triggering episodes
    total_evaluations: int           # total episodes executed
    faults_over_time: List[int]      # cumulative faults at each eval step
    generation_history: List[dict]   # per-gen stats

def compare_g0_g1(g0_results: List[SearchResult],
                  g1_results: List[SearchResult]) -> dict:
    """Compare random (G0) vs danger-guided (G1) search.
    Returns:
        dict with:
          - time_to_first_fault: (g0_mean, g0_std, g1_mean, g1_std)
          - total_faults: (g0_mean, g1_mean, p_value)
          - fault_rate_curve: for plotting
    """
    ...
```

---

## 8. Configuration Structure

```python
# experiments/configs/default.py

from easydict import EasyDict

config = EasyDict(dict(
    seed=42,
    env=dict(
        env_id='CartPole-v1',
        max_steps=500,
    ),
    pets=dict(
        ensemble=dict(
            ensemble_size=5,
            elite_size=3,
            hidden_size=200,
            learning_rate=1e-3,
            holdout_ratio=0.2,
            max_epochs_since_update=5,
        ),
        planner=dict(
            type='random_shooting',   # or 'cem'
            horizon=15,
            num_candidates=500,
            # CEM-specific:
            num_elites=50,
            num_iterations=5,
        ),
        training=dict(
            num_seed_episodes=20,     # random episodes before training
            num_train_epochs=50,
            episodes_per_epoch=5,
        ),
        buffer=dict(
            replay_buffer_size=100_000,
        ),
    ),
    features=dict(
        names=['episode_length', 'max_pole_angle', 'max_cart_position',
               'ensemble_disagreement', 'action_entropy', 'terminal_velocity'],
    ),
    scorer=dict(
        type='ebm',   # ExplainableBoostingClassifier
    ),
    search=dict(
        pop_size=200,
        num_generations=20,
        crossover_prob=0.75,
        mutation_prob=0.30,
        tournament_size=10,
        mutation_noise_range=(0.95, 1.05),
        crossover_match_threshold=0.5,  # obs-space distance
    ),
    evaluation=dict(
        num_runs=10,          # repeat for statistical significance
        budget_per_run=2000,  # max episodes per search run
    ),
))
```

---

## 9. Risks & Mitigations

### 9.1 Technical Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **PETS on discrete actions** — CEM is designed for continuous spaces; CartPole has 2 actions | Core blocker | Use random-shooting planner (Option B) as default; 2-action space makes enumeration tractable |
| **PETS training instability** — ensemble may overfit on small CartPole state space | Poor world model → bad planner | Early stopping with holdout loss (already in DI-engine's MBPO); monitor ensemble disagreement on held-out data |
| **State restore in CartPole** — gymnasium may not support arbitrary state setting | Blocks mutation/crossover re-execution | CartPole-v1 exposes `env.unwrapped.state` (tuple of 4 floats); set it directly + reset step counter |
| **Crossover matching degeneracy** — observation-space proximity may rarely find matches in 4D | Most crossovers fall back to random-point | Increase distance threshold or use cosine similarity; monitor crossover success rate |
| **danger_score quality** — EBM may not learn a useful signal from 6 features | G1 no better than G0 (negative result) | This IS the hypothesis under test; ensure features are well-chosen; inspect EBM shape functions for sanity |
| **Evaluation budget** — if faults are very rare, both G0 and G1 may find zero faults | Inconclusive experiment | First verify that faults exist in PETS CartPole (ablation); if needed, make task harder (tighter bounds, shorter max_steps) |

### 9.2 Dependency Risks

| Risk | Status | Mitigation |
|---|---|---|
| `corels` fails to build with pip ≥ 24 | **Resolved** | Install corels before upgrading pip (see `post-create.sh`) |
| DI-engine hard-pins `numpy<2`, `setuptools<=66.1.1`, `gym==0.25.1` | **Resolved** | Install DI-engine with `--no-deps`; only pull minimal runtime deps |
| DI-engine import may fail at runtime for deeper modules | **Possible** | If a specific DI-engine module needs more deps, add them incrementally. `import ding` itself works fine |
| `gymnasium` vs `gym` API differences in env wrappers | **Low** | We use gymnasium directly; DI-engine's `DingEnvWrapper` supports both |

---

## 10. Implementation Order (suggested)

```
Phase 0: Foundation
  ├── 0.1  Config system (configs/default.py)
  ├── 0.2  Episode data structure (utils/episode.py)
  └── 0.3  Env wrapper with state save/restore (utils/env_wrapper.py)

Phase 1: PETS Agent
  ├── 1.1  EnsembleDynamics wrapping DI-engine's EnsembleModel
  ├── 1.2  RandomShootingPlanner (start simple)
  ├── 1.3  PETSAgent: train loop + act()
  └── 1.4  Verify: PETS solves CartPole-v1 (avg reward > 400)

Phase 2: Feature + Scorer
  ├── 2.1  FeatureExtractor (F1–F6)
  ├── 2.2  Collect labeled dataset: run PETS many times, label faults
  ├── 2.3  DangerScorer (train EBM)
  └── 2.4  Sanity-check: EBM AUC > 0.6 on held-out episodes

Phase 3: Search
  ├── 3.1  Candidate + Archive
  ├── 3.2  Genetic operators (selection, crossover, mutation)
  ├── 3.3  GeneticSearch main loop
  └── 3.4  Verify: search runs end-to-end, finds some faults

Phase 4: Evaluation
  ├── 4.1  Random search baseline (G0)
  ├── 4.2  Guided search (G1)
  ├── 4.3  Statistical comparison
  └── 4.4  Figures + tables
```
