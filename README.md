# udacity-deep-reinforcement-learning

> **Archived.** This is a 2020 course/learning project and is no longer maintained. The dependencies in the `requirements.txt` files are pinned to 2018-era versions (for example `tensorflow==1.15.2`, `torch==0.4.0`, `protobuf==3.5.2`) that have known security vulnerabilities (flagged by Dependabot). They are kept only for reproducibility and should not be installed on machines that are exposed to untrusted input or networks.

Projects from the Udacity Deep Reinforcement Learning Nanodegree. Each project folder has its own README with the problem description, results and details. The trained agents use the Unity ML-Agents environments provided by Udacity.

## Projects

| Folder | Environment | Method | Result (from the project report) |
|---|---|---|---|
| `p1_navigation/` | Banana (Unity), 37-dimensional state, 4 discrete actions | DQN and Double DQN | Double DQN reached an average score of +13 in 360 episodes |
| `p2_continuous_control/` | Reacher (Unity), 33-dimensional state, 4 continuous actions | DDPG (actor-critic) | Reward of +32 reached in 198 episodes |
| `p3_collaboration_competition/` | Tennis (Unity), two agents, 8-dimensional state | MADDPG (multi-agent DDPG) | Average score of +0.5 reached in 2585 episodes |

Each project folder contains the notebook(s), agent and model source code, saved checkpoints (`*.pth`), the report (`Report.md`, plus `Report.pdf` in `p1_navigation/`) and result plots or animations in `results/` or `images/`.

## Unity environment builds (not included)

The Unity environment builds were removed from this repository to keep it small. They are binaries provided by Udacity, not source code. To run the projects, download the build for your operating system from the links in the project README:

- `p1_navigation/README.md` (Banana)
- `p2_continuous_control/README.md` (Reacher)
- `p3_collaboration_competition/README.md` (Tennis)

Unzip the build into the project folder. The notebooks and scripts expect these paths:

- `p1_navigation/Banana_Linux/Banana.x86_64` (the notebook uses `/data/Banana_Linux_NoVis/Banana.x86_64`, so adjust the path to your setup)
- `p2_continuous_control/Reacher_Linux_NoVis/Reacher.x86_64` (used by `run_ddpg.py`)
- `p3_collaboration_competition/Tennis.ipynb` expects a Tennis build at `/data/Tennis_Linux_NoVis/Tennis`

The download links may no longer be available. The Udacity deep-reinforcement-learning GitHub repository (https://github.com/udacity/deep-reinforcement-learning) is the reference for the environment setup and the `unityagents` Python package.

## Tech stack

Python, Jupyter Notebook, PyTorch (the 2018 version pinned in `requirements.txt`), NumPy, matplotlib, and `unityagents` (Unity ML-Agents Python interface). `p2_continuous_control/requirements.txt` also pins TensorFlow 1.15.2. `p3_collaboration_competition/` has no requirements file.

## Running (best effort)

1. Use a Python environment with the packages from the project's `requirements.txt`. The pinned versions are old and may not install on current Python versions.
2. Install `unityagents` as described in the Udacity deep-reinforcement-learning repository.
3. Download and unzip the Unity build for your OS into the project folder (see above).
4. Open the notebook for the project and run it, or see the project README for how to load a saved checkpoint and watch a trained agent. Example: `p2_continuous_control/run_ddpg.py` trains the DDPG agent directly.

These steps were not re-run as part of the archiving cleanup.

## Structure

```
p1_navigation/            DQN / Double DQN agent, notebooks, checkpoints dqn.pth and ddqn.pth
p2_continuous_control/    DDPG agent, run_ddpg.py, notebook, checkpoints checkpoint_actor.pth and checkpoint_critic.pth
p3_collaboration_competition/   Multi-agent DDPG, Tennis notebook, checkpoint_group.pth
```
