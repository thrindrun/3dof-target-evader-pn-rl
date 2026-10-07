# RL Evader Branch

This branch trains the target aircraft with PPO while the missile remains a
fixed Vector-PN controller using the existing `air_to_air_3dof_core.py` physics.

## Files

- `air_to_air_3dof_evader.py`: Gymnasium environment, observation, action,
  reward, episode result and plotting.
- `training/train_evader.py`: single training entry point for VS Code and Colab.
- `evaluation/evaluate_evader.py`: deterministic evaluation and optional CSV/JSON.
- `evader_rl_training.ipynb`: clean Colab orchestration notebook.

Place the Python files in the existing training source:

```text
3DOF_Fuze_Gudum_Project/
├── air_to_air_3dof_core.py
└── 3DOF_Fuze_Gudum_Training_Source/
    ├── air_to_air_3dof_evader.py
    ├── training/
    │   └── train_evader.py
    └── evaluation/
        └── evaluate_evader.py
```

## VS Code setup

From `3DOF_Fuze_Gudum_Training_Source`:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-evader.txt
```

Stage E1:

```powershell
python training/train_evader.py `
  --stage evader_e1_fixed_pn5g `
  --steps 100000 `
  --radius 10 `
  --pn-navigation-constant 4 `
  --pn-max-command-g 5 `
  --max-heading-rate-deg-s 16 `
  --max-flight-path-rate-deg-s 6 `
  --actuator-time-constant-s 0.25 `
  --learning-rate 3e-4 `
  --action-std 0.35 `
  --n-envs 4 `
  --eval-episodes 1 `
  --eval-frequency 10000 `
  --checkpoint-frequency 25000 `
  --device cpu
```

Evaluate against the E1 missile:

```powershell
python evaluation/evaluate_evader.py `
  --model models/evader/evader_e1_fixed_pn5g/best_model.zip `
  --episodes 1 `
  --radius 10 `
  --pn-navigation-constant 4 `
  --pn-max-command-g 5
```

## Curriculum

The nominal 30 g PN missile intercepts all tested constant target commands in
the initial geometry. E1 therefore starts with the same PN law but a fixed 5 g
command limit, where a feasible evasive trajectory exists. Continue the best
policy through fixed stages:

```text
5 g -> 8 g -> 10 g -> 15 g -> 30 g
```

For continuation, give the previous `best_model.zip` with `--source`, reduce the
learning rate (for example to `1e-5`) and reduce `--action-std` (for example to
`0.10`). Add `--randomize` only after the fixed 30 g case is working.

The missile guidance method remains Vector PN throughout. Only its fixed command
limit changes between curriculum stages.
