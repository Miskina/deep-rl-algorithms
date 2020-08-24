# Deep Reinforcement Learning Algorithms #
Implementations of some ***Deep Reinforcement Learning Algorithms*** in PyTorch and OpenAI Gym frameworks.
Some of the algorithms are still under development and may have some bugs.

## Requirements ##
- Python 3.7.3
- PyTorch 1.4
- OpenAI Gym 0.17.1
- NumPy 1.17.2

Full pip state (pip freeze) specified in [requirements](requirements.txt) file.

## Files ##
The files containing the actual implementations are named after the algorithm's abbreviation, while other Python files contain classes and functions used in almost all implementations.

### Core ###
The [core](core.py) file contains generic actor and critic implementations used in multiple algorithms. It also contains a helper function for creating a neural network model.

### Gym Wrappers ###
Multiple wrappers for OpenAI Gym environemnts, primarily used in the *Jupyter Notebooks* for the PPO Pong implementation. Even though some are redundant, they are in the [gym_wrappers](gym_wrappers.py) and [pong_wrappers](pong_wrappers.py) files.

### Current Algorithm Implementations ###
The repository currently includes implementations for:
- [Deep Q-Network](dqn.py)(DQN)
- Policy Gradient - in [pg_2](pg_2.py) and [policy_gradient](policy_gradient.py)
- [Trust Region Policy Optimization](trpo.py) (TRPO)
- Proximal Policy Optimization(PPO) - in [ppo](ppo.py) and [ppo_2](ppo_2.py)

### Python Notebooks ###
The current notebooks ([ppo_pong](ppo_pong.ipynb) and [ppo_pong2](ppo_pong2.ipynb)) implement training for OpenAI Gym
***Pong*** environments.


