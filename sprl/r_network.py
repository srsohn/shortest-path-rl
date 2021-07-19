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

"""R-network and some related functions to train R-networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import gin
from tensorflow import keras
from third_party.keras_resnet import models

@gin.configurable
class RNetwork(object):
  """Encapsulates a trained R network."""
  def __init__(self, input_shape, learning_rate=0.0001, random_state_predictor=False):
    """Inits the RNetwork.

    Args:
      input_shape: (height, width, channel) => being normalized by /255. inside build_siamese_network function!!!
    """
    self._random_state_predictor = random_state_predictor
    self._set_network(input_shape)

    self._r_network.compile(
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1), optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

    self.count = 0

  def _set_network(self, input_shape):
    (self._r_network, self._embedding_network,
     self._similarity_network) = models.ImageResnetBuilder.build_siamese_resnet_18(input_shape)

  def embed_observation(self, x):
    """Embeds an observation.

    Args:
      x: batched input observations. Expected to have the shape specified when
         the RNetwork was contructed (plus the batch dimension as first dim).

    Returns:
      embedding, shape [batch, models.EMBEDDING_DIM]
    """
    return self._embedding_network.predict(x)

  def embedding_similarity(self, k_prev_emb, state):
    """Computes the similarity between two embeddings.

    Args:
      x: batch of the first embedding. Shape [batch, models.EMBEDDING_DIM].
      y: batch of the first embedding. Shape [batch, models.EMBEDDING_DIM].

    Returns:
      Similarity probabilities. 1 means very similar according to the net.
      0 means very dissimilar. Shape [batch].
    """
    similarities = self._similarity_network.predict([k_prev_emb, state],batch_size=256)
    return similarities

class CustomTensorBoard(keras.callbacks.Callback):
  def __init__(self):
      super().__init__()
      self.train_accuracy = 0
      self.valid_accuracy = 0

  def on_batch_end(self, batch, logs={}, is_epoch=False):
    pass

  def on_epoch_end(self, epoch, logs={}):
    self.train_accuracy = logs['acc']
    if 'val_acc' in logs.keys():
      self.valid_accuracy = logs['val_acc']
    else:
      self.valid_accuracy = 0
