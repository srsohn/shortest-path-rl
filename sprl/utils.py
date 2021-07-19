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

"""A few utilities for episodic curiosity.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import tensorflow as tf

from absl import flags
from third_party.baselines import logger

FLAGS = flags.FLAGS

def _unflatten(flat_data, batch_size):
  # [B*N x ...] --> [B x N x ...]
  return flat_data.reshape( [batch_size, -1] + list(flat_data.shape[1:]))

def _flatten(data):
  # [B x N x ...] --> [B*N x ...]
  return data.reshape( [-1] + list(data.shape[2:]))

def get_frame(env_observation, info):
  """Searches for a rendered frame in 'info', fallbacks to the env obs."""
  info_frame = info.get('frame')
  if info_frame is not None:
    return info_frame
  return env_observation

def dump_flags_to_file(filename):
  """Dumps FLAGS to a file."""
  with tf.io.gfile.GFile(filename, 'w') as output:
    output.write('\n'.join([
        '{}={}'.format(flag_name, flag_value)
        for flag_name, flag_value in FLAGS.flag_values_dict().items()
    ]))

def dump_dicts_to_file(filename, dicts):
  """Dumps dicts to a file."""
  with tf.io.gfile.GFile(filename, 'a+') as output:
    output.write('\n')
    output.write('\n'.join([
        '{}={}'.format(name, value)
        for name, value in dicts.items()
    ]))

def load_keras_model(path):
  """Loads a keras model from a h5 file path."""
  # pylint:disable=unreachable
  return tf.keras.models.load_model(path, compile=True)

def dump_pickle(file_path, pickle_data):
    with open(file_path, 'wb') as f:
      pickle.dump(pickle_data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
      pickle_data = pickle.load(f)
    return pickle_data

class Summary_logger():
  def __init__(self, log_dir):
    self.log_dir = log_dir
    self.summary_writer = tf.summary.FileWriter(log_dir)

  def logkv(self, k, v):
    logger.logkv(k, v)
    summary = tf.Summary(value=[tf.Summary.Value(tag=k,
                  simple_value=v)])
    self.summary_writer.add_summary(summary, self.step)

  def set_step(self, step):
    self.step = step

  def dumpkvs(self):
    logger.dumpkvs()
    self.summary_writer.flush()

  def close(self):
    self.summary_writer.close()
