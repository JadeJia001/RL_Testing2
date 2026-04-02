# RL Testing: PETS + EBM Danger-Score Guided Search

Can a learned **danger score** guide a search algorithm to find failure-inducing episodes faster than random testing?

This project implements a complete pipeline to answer this question using:
- **PETS** (Probabilistic Ensemble Trajectory Sampling) as the RL agent under test
- **EBM** (Explainable Boosting Machine) as the danger scorer
- **STARLA-adapted genetic algorithm** as the guided search strategy
- **CartPole-v1** as the test environment

## Experimental Goal

**Goal 1**: Prove that `danger_score = EBM(F1...F6)` can independently guide a genetic search to find failure-inducing episodes faster than random testing.

- Fitness: **only** `danger_score` -- no reward, no fault probability, no certainty
- No perturbation channels (E1-E4)
- Search framework: adapted from STARLA's multi-objective genetic algorithm

## Pipeline Overview

```
Phase 0: Seed Data
  PETS Agent --> train on CartPole --> Trained Ensemble + Planner

Phase 1: Feature + Scorer
  Episodes --> extract F1-F6 --> Train EBM --> danger_score
                               --> Train CORELS --> interpretable rules

Phase 2: Guided Search (G1) vs Random Baseline (G0)
  G1: GA evolves initial-state perturbations to maximize danger_score
  G0: Random perturbations with same budget

Phase 3: Evaluation
  Compare G0 vs G1: fault rate, cumulative faults, efficiency curves
```

## Project Structure

```
RL_Testing2/
├── experiments/
│   ├── configs/
│   │   └── pets_cartpole.py          # Hyperparameters (EasyDict)
│   ├── agents/
│   │   ├── ensemble_model.py         # Probabilistic ensemble dynamics (5 members, 2x64)
│   │   ├── cem_planner.py            # CEM planner (discrete Bernoulli, warm-start, TS-1)
│   │   └── pets_agent.py             # PETS training loop + rollout
│   ├── features/
│   │   └── feature_extractor.py      # 6 danger-relevant features per timestep
│   ├── models/
│   │   ├── train_ebm.py              # Train EBM classifier on features
│   │   ├── train_corels.py           # Train CORELS rule list (interpretable)
│   │   ├── danger_scorer.py          # Wrap EBM for GA fitness (logit transform)
│   │   └── validate_consistency.py   # Cross-validate EBM vs CORELS agreement
│   ├── search/
│   │   ├── genetic_search.py         # Single-objective GA (STARLA-adapted)
│   │   └── random_baseline.py        # G0 random search baseline
│   ├── evaluation/
│   │   ├── run_goal1.py              # G0 vs G1 multi-trial experiment
│   │   ├── plot_efficiency.py        # Generate efficiency curves (Figure 1)
│   │   └── generate_table1.py        # Generate CORELS rules (Table 1)
│   ├── utils/
│   │   ├── episode.py                # Episode dataclass + ReplayBuffer
│   │   └── env_wrapper.py            # CartPole state save/restore
│   ├── train_pets.py                 # Script: train PETS agent
│   ├── collect_episodes.py           # Script: collect 600+ episodes
│   ├── extract_features.py           # Script: extract feature matrix
│   ├── data/                         # Runtime artifacts (checkpoints, episodes, features, results)
│   ├── outputs/                      # Figures and tables
│   └── ARCHITECTURE.md               # Detailed design document
├── DI-engine/                        # OpenDILab DI-engine (submodule, --no-deps)
├── STARLA/                           # STARLA reference implementation (submodule)
├── .devcontainer/                    # Dev container for reproducible environment
└── requirements.txt                  # Python dependencies
```

## Features (F1-F6)

Six per-timestep features capture different aspects of danger:

| ID | Name | Description | Intuition |
|----|------|-------------|-----------|
| F1 | Advantage | Q(s,a) - V(s) | Policy quality: negative = suboptimal action taken |
| F2 | dV | (V_t - V_{t-k}) / k | Value trend: negative = situation deteriorating |
| F3 | Prediction Error | \|\|predicted - actual\|\| | Model reliability: high = poorly modeled region |
| F4 | Uncertainty Accel | d^2U/dt^2 | Worsening speed: positive = uncertainty accelerating |
| F5 | Density | Mean 5-NN distance | Distribution shift: high = far from training data |
| F6 | SNR | \|V\| / (U + eps) | Decision confidence: low = noisy decisions |

## Current Results

### Phase 1: PETS Agent

PETS solves CartPole-v1 (avg reward >= 195) within ~35 iterations using:
- 5-member probabilistic ensemble, 2x64 hidden layers, Swish activation
- Random shooting planner (CEM with 1 iteration, 500 candidates, horizon 10)
- Gaussian NLL loss with logvar clamping, holdout-based early stopping

![PETS Training Curve](experiments/outputs/pets_training_curve.png)

### Phase 2: Episode Collection

600 episodes collected with 20% random-action perturbation:
- Lightweight planner (100 candidates, horizon 8) for speed
- Additional episodes collected to ensure >= 50 failures
- Reference training states saved for F5 density computation

### Phase 3: EBM Danger Scorer

EBM trained on 6 features with episode-level train/test split (no data leakage):

**EBM Shape Functions** -- each feature's contribution to danger_score:

![EBM Shape Functions](experiments/outputs/ebm_shape_functions.png)

Key findings from the shape functions:
- **F1 (Advantage)**: strong negative correlation -- lower advantage = higher danger
- **F2 (dV)**: strong negative correlation -- declining value = higher danger
- **F3 (Prediction Error)**: positive correlation -- high PE = higher danger
- **F5 (Density)**: lower density (closer to training data) shows higher danger contribution, likely reflecting that the agent's failure modes cluster near the training distribution boundary
- **F6 (SNR)**: low SNR = higher danger, with a sharp transition

### Phase 4: CORELS Interpretable Rules

CORELS extracts human-readable if-then rules explaining danger boundaries:

| Rule | Condition | Prediction | Accuracy |
|------|-----------|------------|----------|
| 1 | IF F5_rho_low AND F2_dV_low AND F1_adv_low | failure | 93.3% |
| 2 | IF F5_rho_low AND F2_dV_low AND NOT F1_adv_low | failure | 84.7% |
| 3 | IF F5_rho_low AND NOT F2_dV_low AND F6_SNR_low | failure | 76.7% |
| 4 | IF F5_rho_low AND NOT F2_dV_low AND NOT F6_SNR_low | failure | 79.5% |
| 5 | IF NOT F5_rho_low AND F2_dV_low AND F3_PE_high | failure | 85.4% |

The dominant pattern: **low density (F5) combined with declining value (F2)** is the strongest predictor of failure.

### Phase 5: G0 vs G1 Search Comparison

Experiment setup:
- Population size: 20, Generations: 20, Budget: 420 episodes per trial
- 5 repeated trials for statistical comparison
- Perturbation sigma: 0.05 (same for G0 and G1)
- Both use CEM planner with NO action noise -- only initial-state perturbation differs

The GA (G1) evolves initial-state perturbations to maximize danger_score (logit-transformed EBM output), while G0 uses random perturbations with the same budget.

## Adaptations from STARLA

| Aspect | STARLA Original | Our Adaptation |
|--------|----------------|----------------|
| Agent | Pre-trained DQN (Stable-Baselines) | PETS (ensemble + CEM planner) |
| Fitness | 3 objectives (reward, confidence, fault prob) | 1 objective: danger_score |
| Selection | NSGA-II preference sort | Single-objective tournament |
| Crossover | Abstract-state matching (Q-value bins) | Uniform crossover on state perturbations |
| Mutation | Cart position only (dim 0) | Any state dimension |
| Re-execution | DQN predict | CEM plan |
| Archive | Objective threshold | Early termination = fault |

## Dependencies

| Component | Purpose |
|-----------|---------|
| PyTorch (CPU) | Ensemble dynamics model |
| gymnasium | CartPole-v1 environment |
| interpret (EBM) | Explainable Boosting Classifier |
| corels | Certifiably Optimal Rule Lists |
| DI-engine (--no-deps) | Reference patterns for ensemble training |
| scikit-learn | NearestNeighbors (F5), DecisionTree fallback |
| numpy, scipy, matplotlib, pandas | Numerics and plotting |

## Getting Started

### Using Dev Container (recommended)

```bash
# Open in VS Code with Dev Containers extension, or:
cd .devcontainer && docker build -t rl-testing ..
docker run -it -v $(pwd)/..:/workspace rl-testing bash
bash .devcontainer/post-create.sh
```

### Running the Pipeline

All scripts should be run from the project root:

```bash
# Phase 1: Train PETS agent (~35 min on CPU)
python -m experiments.train_pets

# Phase 2: Collect episodes (~5 min)
python -m experiments.collect_episodes

# Phase 3: Extract features (~10 min)
python -m experiments.extract_features

# Phase 4a: Train EBM
python -m experiments.models.train_ebm

# Phase 4b: Train CORELS
python -m experiments.models.train_corels

# Phase 4c: Validate EBM/CORELS consistency
python -m experiments.models.validate_consistency

# Phase 5: Run G0 vs G1 experiment
python -m experiments.evaluation.run_goal1

# Generate outputs
python -m experiments.evaluation.plot_efficiency
python -m experiments.evaluation.generate_table1
```

## References

- **PETS**: Chua, K., et al. "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models." NeurIPS 2018.
- **STARLA**: Attaoui, M.O., et al. "Black-box Safety Testing of Reinforcement Learning Agents." 2024.
- **EBM**: Nori, H., et al. "InterpretML: A Unified Framework for Machine Learning Interpretability." 2019.
- **CORELS**: Angelino, E., et al. "Learning Certifiably Optimal Rule Lists for Categorical Data." JMLR 2018.
- **DI-engine**: OpenDILab. https://github.com/opendilab/DI-engine
