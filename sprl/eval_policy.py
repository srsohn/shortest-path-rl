# coding=utf-8
# Copyright 2019 Google LLC.
# Copyright 2021 Shortest Path RL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation of a policy on a GYM environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np

@gin.configurable
class PolicyEvaluator(object):
  """Evaluate a policy on a GYM environment."""

  def __init__(self, vec_env, workdir=None):
    """New policy evaluator.

    Args:
      vec_env: baselines.VecEnv correspond to a vector of GYM environments.
      grayscale: Whether the observation is grayscale or color.
      eval_frequency: Only performs evaluation once every eval_frequency times.
    """
    self._vec_env = vec_env
    self.workdir = workdir

  def evaluate(self, model_step_fn):
    """Evaluate the policy as given by its step function.
    Args:
      model_step_fn: Function which given a batch of observations,
        a batch of policy states and a batch of dones flags returns
        a batch of selected actions and updated policy states.
    """
    # Reset the environments before starting the evaluation.
    dones = [False] * self._vec_env.num_envs
    sticky_dones = [False] * self._vec_env.num_envs
    obs = self._vec_env.reset()
    infos = self._vec_env.get_initial_infos()

    # Evaluation loop.
    total_reward = np.zeros((self._vec_env.num_envs,), dtype=np.float32)

    while not all(sticky_dones):
      actions = model_step_fn(obs, dones, infos)

      # Update the states of the environment based on the selected actions.
      obs, rewards, dones, infos = self._vec_env.step(actions)

      for k in range(self._vec_env.num_envs,):
        if not sticky_dones[k]:
          total_reward[k] += rewards[k]

      sticky_dones = [sd or d for (sd, d) in zip(sticky_dones, dones)]

    print('Average reward: {}, total reward: {}'.format(np.mean(total_reward),
                                                        total_reward))
