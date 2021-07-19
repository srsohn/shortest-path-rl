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

"""Wrapper around a Gym environment to add curiosity reward."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import gin
import gym
import numpy as np
import cv2
import math
import tensorflow as tf
from PIL import Image
from sprl import oracle
from third_party.baselines.common.vec_env import VecEnv
from third_party.baselines.common.vec_env import VecEnvWrapper
from collections import defaultdict

class MovingAverage(object):
  """Computes the moving average of a variable."""

  def __init__(self, capacity=100):
    self._capacity = capacity
    self._history = np.array([0.0] * capacity)
    self._size = 0

  def add(self, value):
    index = self._size % self._capacity
    self._history[index] = value
    self._size += 1

  def mean(self):
    if not self._size:
      return None
    if self._size < self._capacity:
      return np.mean(self._history[0:self._size])
    return np.mean(self._history)


@gin.configurable
class CuriosityEnvWrapper(VecEnvWrapper):
  """Environment wrapper that adds additional curiosity reward."""
  def __init__(self,
               vec_env,
               observation_embedding_fn,
               observation_compare_fn,
               target_image_shape,
               r_network_trainer=None,
               r_net=None,
               embedding_size=None,
               exploration_reward=None,
               scale_task_reward=1.0,
               scale_oracle_reward=0.0,
               similarity_threshold=0.5,
               max_action_distance=5,
               tolerance=0.,
               policy_architecture=None,
               max_total_env_step=1e6,
               gamma=0.99,
               r_network_weights_store_path=None,
               is_env=None,):

    if self._should_postprocess_observation(vec_env.observation_space.shape):
      observation_space_shape = target_image_shape[:]
      observation_space = gym.spaces.Box(
          low=0, high=255, shape=observation_space_shape, dtype=np.uint8)
    else:
      observation_space = vec_env.observation_space
    self.bare_obs_space = observation_space
    VecEnvWrapper.__init__(self, vec_env, observation_space=observation_space)

    self.action_dim = self.action_space.n
    self.r_network_trainer = r_network_trainer
    self.r_net = r_net
    self._embedding_size = embedding_size
    self._observation_embedding_fn = observation_embedding_fn
    self._observation_compare_fn = observation_compare_fn
    self._target_image_shape = target_image_shape

    self._exploration_reward = exploration_reward
    self._scale_task_reward = scale_task_reward
    self._scale_oracle_reward = scale_oracle_reward
    self.policy_architecture = policy_architecture
    self._gamma = gamma
    self.num_envs = self.venv.num_envs
    self.is_env = is_env

    # Oracle reward.
    self._oracles = [oracle.OracleExplorationReward()
                     for _ in range(self.venv.num_envs)]

    # Total number of steps so far per environment.
    self._step_count = 0
    self._total_step_count = 0

    # SPRL
    self._max_action_distance = max_action_distance
    self._tolerance = tolerance
    self._mdporder = max_action_distance + tolerance + 1
    self.max_total_step = max_total_env_step

    self.checkpoint_dir = r_network_weights_store_path
    self.count = 0

  def get_initial_infos(self):
    infos = [dict() for _ in range(self.num_envs)]
    infos[0]['observations'] = self.initial_observations
    return infos

  def resize_observation(self, frame, image_shape, worker_idx, reward=None):
      """Resize an observation according to the target image shape.
      observation : original size <=> frame : resized observation"""
      height, width, target_depth = image_shape
      if frame.shape == (height, width, target_depth):
        return frame
      if frame.shape[-1] != 3 and frame.shape[-1] != 1:
        raise ValueError(
            'Expecting color or grayscale images, got shape {}: {}'.format(
                frame.shape, frame))
      if frame.shape[-1] == 3 and target_depth == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

      frame = cv2.resize(frame, (width, height),
                        interpolation=cv2.INTER_AREA)

      # OpenCV operations removes the last axis for grayscale images.
      # Restore the last axis.
      if len(frame.shape) != 3:
        frame = frame[:, :, np.newaxis]
      return frame

  def _should_postprocess_observation(self, obs_shape):
    # Only post-process observations that look like an image.
    return len(obs_shape) >= 3

  def _postprocess_observation(self, observations, rewards=None):
    if not self._should_postprocess_observation(observations[0].shape):
      return observations
    post_obs = observations
    return post_obs

  def step_wait(self):
    """Overrides VecEnvWrapper.step_wait."""
    self._step_count += 1
    self._total_step_count += self.num_envs

    # 1. rollout environment one step
    observations, rewards, dones, infos = self.venv.step_wait()

    obs_emb = None

    # 2. Compute oracle bonus
    if 'oracle' in self._exploration_reward:
      assert self._exploration_reward in ['oracle']
      bonus_rewards = self._compute_oracle_reward(infos, dones)
      rewards = (self._scale_task_reward * rewards + self._scale_oracle_reward * bonus_rewards['oracle_bonus'])

    # 3. Post-processing on the obs.
    postprocessed_observations = self._postprocess_observation(observations, rewards)
    infos[0]['observations'] = observations

    return postprocessed_observations, rewards, dones, infos

  def _compute_oracle_reward(self, infos, dones):
    if np.any(infos[0]['position'] == None):
      bonus = [infos[k]['oracle_bonus']
        for k in range(self.venv.num_envs)]
      bonus = np.array(bonus)
    else:
      bonus = [
          self._oracles[k].update_position(infos[k]['position'])
          for k in range(self.venv.num_envs)]
      for k in range(self.venv.num_envs):
        if dones[k]:
          self._oracles[k].reset()

    bonus = np.array(bonus)
    bonus_dict = {'oracle_bonus': bonus}
    return bonus_dict

  def reset(self):
    """Overrides VecEnvWrapper.reset."""
    observations = self.venv.reset()
    self.initial_observations = observations
    postprocessed_observations = self._postprocess_observation(observations)
    ##### SPRL or ECO #####
    if self._max_action_distance > 0:
      self._kstep_obs_memory_index = 0
      self._kstep_emb_memory_index = 0

    # Initial infos
    return postprocessed_observations
