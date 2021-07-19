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

"""Set of functions used to train a R-network."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import gin
import cv2
import random
import statistics
import numpy as np
import tensorflow as tf

from tensorflow import keras
from collections import deque
from PIL import Image, ImageFont, ImageDraw
from sprl.constants import Const
from sprl import keras_checkpoint
from sprl.utils import dump_pickle, load_pickle
from sprl.r_network import CustomTensorBoard

def generate_positive_example(anchor,
                              next_anchor):
  """Generates a close enough pair of states."""
  first = anchor
  second = next_anchor
  return first, second

@gin.configurable
def generate_negative_example(anchor,
                              len_episode_buffer,
                              neg_sample_threshold,
                              max_action_distance,
                              neg_sample_upper_ratio=0.1):
  """Generates a far enough pair of states."""
  assert anchor < len_episode_buffer
  min_index = anchor + neg_sample_threshold
  if min_index >= len_episode_buffer - 1:
    return anchor, None

  max_index = min(min_index + int(neg_sample_upper_ratio * neg_sample_threshold), len_episode_buffer-1)
  index = random.randint(min_index, max_index)

  return anchor, index


def compute_next_anchor(anchor):
  """Computes the buffer position for the next training example."""
  return anchor + random.randint(1, 5) + 1

@gin.configurable
def sample_triplet_data(buffer_len,
                        positive_limit,
                        neg_sample_threshold,
                        max_action_distance,
                        neg_pos_ratio=1):
  first_second_label = []
  anchor = 0
  while True:
    positive_example_candidate = (
        anchor + random.randint(1, positive_limit))
    next_anchor = anchor + random.randint(2, 6)

    if next_anchor is None or buffer_len is None or positive_example_candidate is None:
      raise ValueError("Training data for RNet is empty.")
    if (next_anchor >= buffer_len or
        positive_example_candidate >= buffer_len):
      break

    positive = positive_example_candidate
    if positive is None:
      break
    assert anchor < positive
    first_second_label.append((anchor, positive, 1))

    for _ in range(neg_pos_ratio):
      _, negative = generate_negative_example(anchor=anchor,
                                              len_episode_buffer=buffer_len,
                                              neg_sample_threshold=neg_sample_threshold,
                                              max_action_distance=max_action_distance)
      if negative is None:
        break
      assert anchor < negative
      first_second_label.append((anchor, negative, 0))

    anchor = next_anchor
  return first_second_label

@gin.configurable
def create_training_data(episode_buffer,
                         max_action_distance,
                         neg_sample_threshold):
  """Samples intervals and forms pairs."""
  buffer_len = len(episode_buffer)
  first_second_label = sample_triplet_data(buffer_len=buffer_len,
                                           positive_limit=max_action_distance,
                                           neg_sample_threshold=neg_sample_threshold,
                                           max_action_distance=max_action_distance)

  x1 = []
  x2 = []
  labels = []

  for first, second, label in first_second_label:
    x1.append(episode_buffer[first])
    x2.append(episode_buffer[second])
    labels.append([label])
  return x1, x2, labels

@gin.configurable
class RNetworkTrainer(object):
  """Train a R network in an online way."""
  def __init__(self,
               r_model,
               observation_history_size=20000,
               num_envs=None,
               buffer_size=20000,
               num_epochs=2,
               train_size=40000,
               checkpoint_dir=None,
               max_action_distance=None,
               tolerance=None,
               neg_sample_adder=None,
               obs_emb_fn=None,
               is_env=None):
    # The buffer_size is assumed to be the same as the history size
    # for invalid negative values.
    if buffer_size < 0:
      buffer_size = observation_history_size

    self._r_model = r_model
    self._num_envs = num_envs
    self._buffer_size = buffer_size // num_envs
    self._batch_size = 128
    self._num_epochs = num_epochs
    self.train_size = train_size
    self._max_action_distance = max_action_distance
    self._tolerance = tolerance
    self._mdporder = int(self._max_action_distance + self._tolerance + 1)
    self._neg_sample_adder = neg_sample_adder
    self._min_accuracy = 0.75
    self.obs_emb_fn = obs_emb_fn

    self._tb_callback = CustomTensorBoard()

    # Keeps track of the last N observations.
    # Those are used to train the R network in an online way.
    observation_history_size = observation_history_size // num_envs
    self._fifo_observations = [None] * observation_history_size
    self._fifo_actions = [None] * observation_history_size
    self._fifo_rewards = [None] * observation_history_size
    self._fifo_next_dones = [None] * observation_history_size
    self._fifo_bonuses = [None] * observation_history_size
    self._fifo_topdown_obses = [None] * observation_history_size
    self._fifo_index = 0
    self._fifo_count = 0

    # For is_ready
    self._accuracy_buffer = deque(maxlen=20)
    self._algo_ready = False

    # Used to save checkpoints.
    self._current_epoch = 0
    self._checkpointer = None
    if checkpoint_dir is not None:
      self._checkpoint_dir = checkpoint_dir
      checkpoint_period_in_epochs = self._num_epochs * 100
      self._checkpointer = keras_checkpoint.GFileModelCheckpoint(
          os.path.join(checkpoint_dir, 'r_network_weights.{epoch:05d}.h5'),
          save_summary=False,
          save_weights_only=False,
          period=checkpoint_period_in_epochs)
      self._checkpointer.set_model(self._r_model)

    else:
      self._checkpoint_dir = os.getcwd()+'/r_net_online_training_log'
      raise NotImplementedError

    self._step = 0
    self.train_data_size = 0

    self.is_env = is_env

  def train_ready(self):
    return self._fifo_count > (self._buffer_size) // 2

  def algo_ready(self):
    return self._algo_ready

  def get_train_accuracy(self):
    return self._tb_callback.train_accuracy

  def get_valid_accuracy(self):
    return self._tb_callback.valid_accuracy

  def get_negative_threshold(self):
    return self._neg_sample_adder

  def _add_to_buffer(self, obs, actions, rewards, next_dones, batch=False):
    if batch:
      assert type(obs)==list and len(obs[0].shape) == 4
      batch_size = len(obs)
      if batch_size + self._fifo_index >= len(self._fifo_observations):
        batch_size_first = len(self._fifo_observations) - self._fifo_index
        self._fifo_observations[self._fifo_index:] = obs[:batch_size_first]
        self._fifo_actions[self._fifo_index:] = actions[:batch_size_first]
        self._fifo_rewards[self._fifo_index:] = rewards[:batch_size_first]
        self._fifo_next_dones[self._fifo_index:] = next_dones[:batch_size_first]

        batch_size_second = batch_size - batch_size_first
        self._fifo_observations[:batch_size_second] = obs[batch_size_first:]
        self._fifo_actions[:batch_size_second] = actions[batch_size_first:]
        self._fifo_rewards[:batch_size_second] = rewards[batch_size_first:]
        self._fifo_next_dones[:batch_size_second] = next_dones[batch_size_first:]
        self._fifo_index = batch_size_second
      else:
        self._fifo_observations[self._fifo_index:self._fifo_index+batch_size] = obs
        self._fifo_actions[self._fifo_index:self._fifo_index+batch_size] = actions
        self._fifo_rewards[self._fifo_index:self._fifo_index+batch_size] = rewards
        self._fifo_next_dones[self._fifo_index:self._fifo_index+batch_size] = next_dones
        self._fifo_index += batch_size
      self._fifo_count += batch_size
    else:
      assert len(obs.shape) == 4
      self._fifo_observations[self._fifo_index] = obs
      self._fifo_actions[self._fifo_index] = actions
      self._fifo_rewards[self._fifo_index] = rewards
      self._fifo_next_dones[self._fifo_index] = next_dones
      self._fifo_index = (
          (self._fifo_index + 1) % len(self._fifo_observations))
      self._fifo_count += 1

  def update_buffer(self, observations, actions, rewards, next_dones):
    self._add_to_buffer(observations, actions, rewards, next_dones, batch=True)

  def save_buffer(self):
    dump_pickle('random_rnet_buffer.pkl', {'obs':self._fifo_observations,'a':self._fifo_actions,'r':self._fifo_rewards,'d':self._fifo_next_dones})

  def load_buffer(self):
    buffer = load_pickle('random_rnet_buffer.pkl')
    self._fifo_observations = buffer['obs']
    self._fifo_actions = buffer['a']
    self._fifo_rewards = buffer['r']
    self._fifo_next_dones = buffer['d']
    self._fifo_count = len(self._fifo_observations)

  def train(self, update, nupdates):
    if self._max_action_distance > 0:
      assert self._max_action_distance != None
      history_observations, history_actions, history_rewards, history_next_dones = self._get_flatten_history()

      # Train R-net
      if self.obs_emb_fn != None:
        history_shape = np.shape(history_observations)[:2]
        obs_shape = np.shape(history_observations[0][0])
        observation_history_size = history_shape[0]
        num_envs = history_shape[1]
        num_batch = 12
        observation_batch_size = int(observation_history_size*num_envs / num_batch)
        history_observations = np.reshape(history_observations, (observation_history_size*num_envs,)+obs_shape)
        history_obs_embs = []
        for batch_idx in range(num_batch):
          history_obs_embs.append(self.obs_emb_fn(history_observations[batch_idx*observation_batch_size:(batch_idx+1)*observation_batch_size]))
        history_observations = np.reshape(history_obs_embs, (observation_history_size,num_envs,-1))
      self._train(history_observations, history_actions, history_rewards, history_next_dones, self._max_action_distance)

      self._accuracy_buffer.append(self._tb_callback.train_accuracy)

      # If accuracy saturates, start applying R-net
      if len(self._accuracy_buffer) == self._accuracy_buffer.maxlen and np.mean(self._accuracy_buffer) > self._min_accuracy:
        self._algo_ready = True

  def _get_flatten_history(self):
    """Convert the history given as a circular fifo to a linear array."""
    if self._fifo_count < len(self._fifo_observations):
      return (self._fifo_observations[:self._fifo_count],
              self._fifo_actions[:self._fifo_count],
              self._fifo_rewards[:self._fifo_count],
              self._fifo_next_dones[:self._fifo_count])

    # Reorder the indices.
    history_observations = self._fifo_observations[self._fifo_index:]
    history_observations.extend(self._fifo_observations[:self._fifo_index])

    history_actions = self._fifo_actions[self._fifo_index:]
    history_actions.extend(self._fifo_actions[:self._fifo_index])

    history_rewards = self._fifo_rewards[self._fifo_index:]
    history_rewards.extend(self._fifo_rewards[:self._fifo_index])

    history_next_dones = self._fifo_next_dones[self._fifo_index:]
    history_next_dones.extend(self._fifo_next_dones[:self._fifo_index])

    return history_observations, history_actions, history_rewards, history_next_dones

  def _split_history(self, observations, rewards, next_dones):
    """Returns some individual trajectories."""
    if len(observations) == 0:  # pylint: disable=g-explicit-length-test
      return []

    # Number of environments that generated "observations",
    # and total number of steps.
    nenvs = len(next_dones[0])
    nsteps = len(next_dones)

    # Starting index of the current trajectory.
    start_index = [0] * nenvs
    trajectories = []
    returns = []
    for k in range(nsteps):
      for n in range(nenvs):
        if (next_dones[k][n]) or (rewards[k][n]>0) or (k == nsteps - 1):
          next_start_index = k + 1
          time_slice = observations[start_index[n]:next_start_index]
          trajectories.append([obs[n] for obs in time_slice])
          start_index[n] = next_start_index
    return trajectories

  def _prepare_data(self, observations, rewards, next_dones, max_action_distance, mode, test_adder=1):
    """Generate the positive and negative pairs used to train the R network."""
    all_x1 = []
    all_x2 = []
    all_labels = []
    trajectories = self._split_history(observations, rewards, next_dones)

    for trajectory in trajectories:
      if mode == 'train':
        x1, x2, labels = create_training_data(
            trajectory, max_action_distance,
            neg_sample_threshold=int(self._max_action_distance + self._neg_sample_adder))
      elif mode == 'valid':
        x1, x2, labels = create_training_data(
            trajectory, max_action_distance,
            neg_sample_threshold=self._max_action_distance + int(self._neg_sample_adder/2))
      elif mode == 'test':
        x1, x2, labels = create_training_data(
            trajectory, max_action_distance,
            neg_sample_threshold=self._max_action_distance + test_adder)
      all_x1.extend(x1)
      all_x2.extend(x2)
      all_labels.extend(labels)

    return all_x1, all_x2, all_labels

  def _shuffle(self, x1, x2, labels):
    sample_count = len(x1)
    assert len(x2) == sample_count
    assert len(labels) == sample_count
    permutation = np.random.permutation(sample_count)
    x1 = [x1[p] for p in permutation]
    x2 = [x2[p] for p in permutation]
    labels = [labels[p] for p in permutation]
    return x1, x2, labels

  def _train(self, history_observations, history_actions, history_rewards, history_next_dones, max_action_distance):
    del history_actions
    """Do one pass of training of the R-network."""
    x1, x2, labels = self._prepare_data(history_observations, history_rewards, history_next_dones, max_action_distance, mode='train')
    x1, x2, labels = self._shuffle(x1, x2, labels)

    train_len = min(self.train_size, len(x1))
    if train_len >= self._batch_size:
      x1_train, x2_train, labels_train = (x1[:train_len], x2[:train_len], labels[:train_len])
      self.train_data_size = len(x1_train)

      online_rnet_history = self._r_model.fit_generator(
          self._generate_batch(x1_train, x2_train, labels_train, tag='train'),
          steps_per_epoch=train_len // self._batch_size,
          epochs=self._num_epochs,
          callbacks=[self._tb_callback],
          verbose=0)

      for _ in range(self._num_epochs):
        self._current_epoch += 1
        if self._checkpointer is not None:
          self._checkpointer.on_epoch_end(self._current_epoch)

  def _generate_batch(self, x1, x2, labels, tag=''):
    """Generate batches of data used to train the R network."""
    while True:
      # Train for one epoch.
      sample_count = len(x1)
      number_of_batches = sample_count // self._batch_size
      for batch_index in range(number_of_batches):
        from_index = batch_index * self._batch_size
        to_index = (batch_index + 1) * self._batch_size
        yield ([np.array(x1[from_index:to_index]),
                np.array(x2[from_index:to_index])],
                np.array(labels[from_index:to_index]),
              )

      # After each epoch, shuffle the data.
      x1, x2, labels = self._shuffle(x1, x2, labels)
