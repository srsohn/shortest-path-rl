# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
from absl import app
from sprl.utils import _flatten, _unflatten

@gin.configurable
class BonusModel(object):
  """Compute bonus.
  Deal with
  - kstep_buffer: for SPRL
  - rnet: for (SPRL, ECO)
  """
  def __init__(self,
               num_envs,
               # Bonus
               bonus_type=(),
               scale_shortest_bonus=0.03,
               # R-net
               emb_dim=0,
               obs_emb_fn=None,
               obs_compare_fn=None,
               tolerance=0.,
               max_action_distance=0,
               bonus_bias=0,
               # SPRL
               use_various_tol=False,
               tol_num=1,
               bonus_thres=90,
               mdpadder=0,
               # ECO
               epmem_size=200,
               scale_eco_bonus=0.03,
               # Environment
               is_env=None,
               ):
    self.emb_dim = emb_dim
    self.num_envs = num_envs
    self.bonus_type = bonus_type
    # R-net
    self.obs_emb_fn = obs_emb_fn
    self.obs_compare_fn = obs_compare_fn
    # SPRL
    self.scale_shortest_bonus = scale_shortest_bonus
    self.batch_step_counter = np.zeros((self.num_envs))
    self.max_action_distance = max_action_distance
    self.mdporder = (max_action_distance + tolerance + 1)
    self.tolerance = tolerance
    self.mdpadder = mdpadder
    # ICM
    self.icm_weight = 0.0
    # ECO
    self._update_epmem_freq = 1
    self.scale_eco_bonus = scale_eco_bonus
    # Environment
    self.is_env = is_env

    self.bonus_bias = bonus_bias

    self._use_sprl = 'sprl' in bonus_type
    self._use_eco = 'eco' in bonus_type
    self._use_icm = 'icm' in bonus_type
    self._use_rnet = self._use_sprl or self._use_eco
    self._use_various_tol = use_various_tol
    self._use_epmem = self._use_eco

    if self._use_sprl:
      self.tol_num = tol_num
      if self._use_various_tol:
        self.stability_num = 20
        self.max_tol_num = max(tol_num,self.stability_num)
      else:
        self.stability_num = 1
        self.max_tol_num = self.stability_num

      various_tol_list = np.array(range(self.max_tol_num))*self.tolerance

      if not self._use_various_tol:
        self.mdporder_vec = self.max_action_distance + self.mdpadder + np.array([1])*self.tolerance + 1
      elif self.stability_num > tol_num:
        self.mdporder_vec = self.max_action_distance + self.mdpadder + np.array(range(self.stability_num)) + 1
      else:
        various_tol_ind = np.where((various_tol_list >= self.stability_num)==True)[0][0]
        self.mdporder_vec = self.max_action_distance + self.mdpadder + np.concatenate((np.array(range(self.stability_num)),various_tol_list[various_tol_ind:]), axis=0) + 1

      self.mdporder_vec = np.expand_dims(self.mdporder_vec, -1)
      self._tol_num = self.mdporder_vec.shape[0]
      self._max_tol = self.max_action_distance + self.mdpadder + self.max_tol_num*tolerance+1
      self._kstep_memory = np.zeros([self._max_tol*self.num_envs] + emb_dim)

    if self._use_epmem:
      self._epmem_size = epmem_size
      self._epmem = np.zeros([epmem_size,self.num_envs] + emb_dim)
      self._epmem_count = np.zeros([self.num_envs], dtype=np.int32)

    # tolerance annealing
    self._bonus_thres = bonus_thres

  @property
  def use_bonus(self):
    return self._use_sprl or self._use_icm or self._use_eco

  def get_bonus(self, mb_obs, mb_dones, prev_mb_rewards, mb_infos, is_rnet_ready):
    # Input: prev_mb_rewards: [256 , Nenv]
    # : bonus = R(s(t+1-mdporder), s(t+1)) * valid_mask  [for dmlab]
    # : valid_flag = 0 if (done or prev_reward)

    if not (self._use_sprl or self._use_eco):
      assert False
    batch_size, num_env = mb_obs.shape[:2]
    assert num_env == self.num_envs
    flat_mb_obs = _flatten(mb_obs)
    flat_mb_embs = self.obs_emb_fn(flat_mb_obs)

    shortest_bonus, raw_shortest_bonus, eco_bonus, raw_eco_bonus = np.zeros_like(prev_mb_rewards), np.zeros_like(prev_mb_rewards), np.zeros_like(prev_mb_rewards), np.zeros_like(prev_mb_rewards)
    if self._use_sprl:
      # 1. Update k-step mask (always)
      flat_mb_kstep_valids = self._update_kstep_valid(mb_dones, prev_mb_rewards)

      # 2. Compute bonus (only if r-net is ready)
      if is_rnet_ready:
        flat_shortest_bonus, raw_flat_shortest_bonus = self._compute_shortest_bonus(flat_mb_embs, mb_infos, mb_dones, flat_mb_kstep_valids)
        shortest_bonus = _unflatten(flat_shortest_bonus, batch_size)
        raw_shortest_bonus = _unflatten(raw_flat_shortest_bonus, batch_size)
        assert prev_mb_rewards.shape == raw_shortest_bonus.shape

      # 3. Store last k-steps for the next time
      self._kstep_memory = flat_mb_embs.copy()[-self._max_tol * self.num_envs:]

    if self._use_eco:
      if is_rnet_ready:
        flat_eco_bonus, raw_flat_eco_bonus = self._compute_eco_bonus(flat_mb_embs, mb_infos, mb_dones)
        eco_bonus = _unflatten(flat_eco_bonus, batch_size)
        raw_eco_bonus = _unflatten(raw_flat_eco_bonus, batch_size)
        assert prev_mb_rewards.shape == raw_eco_bonus.shape

    bonus_dict = {
      'bonus': self.scale_shortest_bonus * shortest_bonus + self.scale_eco_bonus * eco_bonus,
    }
    if self._use_sprl:
      bonus_dict.update(
        {
          'scale_shortest_bonus': self.scale_shortest_bonus,
          'shortest': shortest_bonus,
          'raw_shortest': raw_shortest_bonus,
        }
      )
    if self._use_eco:
      bonus_dict.update(
        {
          'eco': eco_bonus,
          'raw_eco': raw_eco_bonus,
          'scale_eco_bonus': self.scale_eco_bonus,
        }
      )
    return bonus_dict

  def _update_kstep_valid(self, mb_dones, prev_mb_rewards):
    # mb_dones: dones [256, n_env]
    # prev_mb_rewards: rewards [256, n_env]
    batch_size = len(mb_dones)
    num_env = len(mb_dones[0])

    # 1. Compute k-step-mask
    batch_not_dones = ~mb_dones
    batch_prev_no_rewards = prev_mb_rewards == 0
    batch_kstep_valids = np.zeros((self._tol_num, batch_size, num_env))
    for bid, (not_dones, prev_no_rewards) in enumerate(zip(batch_not_dones, batch_prev_no_rewards)):
      self.batch_step_counter += 1
      self.batch_step_counter = self.batch_step_counter * not_dones * prev_no_rewards
      batch_kstep_valids[:, bid, :] = self.batch_step_counter >= self.mdporder_vec
    flat_mb_kstep_valids = batch_kstep_valids.flatten()

    return flat_mb_kstep_valids

  def _compute_shortest_bonus(self, flat_mb_embs, mb_infos, mb_dones, flat_mb_kstep_valids):
    # flat_mb_embs: observation batch [256*n_env x EMB_DIM]
    # return: flat_shortest_bonus [256*n_env,]
    batch_size = len(mb_dones)
    num_env = len(mb_dones[0])
    emb_dim = np.shape(flat_mb_embs)

    # 1. Compute bonus (only if R-net is ready)
    total_mb_emb = np.concatenate((self._kstep_memory, flat_mb_embs), axis=0)
    stack_k_prev_mb_emb = []
    for tol_idx in range(self._tol_num):
      stack_k_prev_mb_emb.append(total_mb_emb[-(batch_size+self.mdporder_vec[tol_idx][0])* self.num_envs : -(self.mdporder_vec[tol_idx][0])* self.num_envs])

    stack_k_prev_mb_emb = np.array(stack_k_prev_mb_emb)

    assert len(np.shape(stack_k_prev_mb_emb)) == 3
    stack_k_prev_mb_emb_sum = stack_k_prev_mb_emb.sum(axis=1).sum(axis=1)

    if stack_k_prev_mb_emb_sum[-1] == 0:
      stack_k_prev_mb_emb_sum_zero = np.where(stack_k_prev_mb_emb_sum==0)[0][0]
    else:
      stack_k_prev_mb_emb_sum_zero = np.shape(stack_k_prev_mb_emb)[0]

    stack_k_prev_mb_emb = stack_k_prev_mb_emb[:stack_k_prev_mb_emb_sum_zero]
    flat_mb_kstep_valids = flat_mb_kstep_valids[:stack_k_prev_mb_emb_sum_zero*batch_size*num_env]

    stack_mb_emb = np.repeat(np.expand_dims(flat_mb_embs,0), stack_k_prev_mb_emb_sum_zero, axis=0)

    # Measure similarity
    if stack_k_prev_mb_emb.shape != stack_mb_emb.shape:
      print('Error! Shape is different!')
      raise ValueError

    flat_stack_mb_emb = stack_mb_emb.reshape(-1, *emb_dim[1:])
    flat_stack_k_prev_mb_emb = stack_k_prev_mb_emb.reshape(-1, *emb_dim[1:])

    similarity_to_previous = self.obs_compare_fn(flat_stack_k_prev_mb_emb, flat_stack_mb_emb).squeeze()

    del stack_k_prev_mb_emb
    del stack_mb_emb
    del flat_stack_k_prev_mb_emb
    del flat_stack_mb_emb
    del mb_infos
    del mb_dones
    bias = self.bonus_bias

    similarity_to_previous = np.reshape(similarity_to_previous * flat_mb_kstep_valids, (stack_k_prev_mb_emb_sum_zero, -1))

    if self._use_various_tol:
      stability_similarity = np.percentile(similarity_to_previous, self._bonus_thres, axis=0, interpolation='nearest')
    else:
      stability_similarity = np.mean(similarity_to_previous, axis=0)

    del similarity_to_previous

    raw_flat_shortest_bonus = bias - stability_similarity
    flat_shortest_bonus = bias - (stability_similarity > 0.5)
    return (flat_shortest_bonus, raw_flat_shortest_bonus)

  def _compute_eco_bonus(self, flat_embs, infos, dones):
    # flat_embs: observation batch [256*n_env x EMB_DIM]
    # return: flat_shortest_bonus [256*n_env,]
    batch_size = len(dones)
    num_env = len(dones[0])
    emb_dim = np.shape(flat_embs)

    assert batch_size % self._update_epmem_freq == 0

    bonus = []
    raw_bonus = []

    for minibatch_idx in range(int(batch_size/self._update_epmem_freq)):
      flat_mb_embs = flat_embs[minibatch_idx*self._update_epmem_freq*num_env:(minibatch_idx+1)*self._update_epmem_freq*num_env]
      mb_infos = infos[minibatch_idx*self._update_epmem_freq:(minibatch_idx+1)*self._update_epmem_freq]
      mb_dones = dones[minibatch_idx*self._update_epmem_freq:(minibatch_idx+1)*self._update_epmem_freq]

      # 1. Compute bonus (only if R-net is ready)
      stack_mb_emb = np.repeat(np.expand_dims(flat_mb_embs,0), self._epmem_size, axis=0)
      flat_stack_mb_emb = stack_mb_emb.reshape(-1, *emb_dim[1:])

      stack_epmem = np.repeat(self._epmem, self._update_epmem_freq, axis=1)
      flat_stack_epmem = stack_epmem.reshape(-1, *emb_dim[1:])

      if flat_stack_epmem.shape != flat_stack_mb_emb.shape:
        print('Error! Shape is different!')
        raise ValueError

      similarity_to_previous = self.obs_compare_fn(flat_stack_epmem, flat_stack_mb_emb).squeeze()
      bias = self.bonus_bias
      # Positive: bonus = 1 - similarity * mask
      # Negative: bonus = - similarity * mask

      similarity_to_previous = np.reshape(similarity_to_previous, (self._epmem_size, self._update_epmem_freq, num_env))

      # 2. Update Episodic Memory
      dones_per_env = np.sum(mb_dones, 0)
      reshaped_mb_embs = flat_mb_embs.reshape((self._update_epmem_freq, num_env, *emb_dim[1:]))
      for env_idx in range(num_env):
        if self._epmem_count[env_idx] == 0:
          similarity_per_env = np.zeros((self._update_epmem_freq))
        elif self._epmem_count[env_idx] == 1:
          similarity_per_env = similarity_to_previous[0,:,env_idx]
        else:
          similarity_per_env = np.percentile(similarity_to_previous[:self._epmem_count[env_idx],:,env_idx], self._bonus_thres, axis=0)

        raw_flat_eco_bonus = bias - similarity_per_env
        flat_eco_bonus = bias - similarity_per_env

        if dones_per_env[env_idx] > 0:
          # reset count & mem
          self._epmem[:,env_idx,:] *= 0
          self._epmem_count[env_idx] = 0
        else:
          similarity_env = (similarity_per_env > 0.5)

          # only update 1 state at most at a time
          if np.mean(similarity_env) < 1:
            dissimilar_index = np.where(similarity_env==0)[0][0]
            embs_env = reshaped_mb_embs[dissimilar_index,env_idx,:]
            if self._epmem_count[env_idx] < self._epmem_size:
              epmem_count = self._epmem_count[env_idx]
            else:
              epmem_count = np.random.randint(self._epmem_size)
            self._epmem[epmem_count,env_idx,:] = embs_env.copy()
            if env_idx == 0:
              pass
            # update memory count
            self._epmem_count[env_idx] += 1
            raw_flat_eco_bonus[dissimilar_index] = (bias - similarity_per_env)[dissimilar_index]
            flat_eco_bonus[dissimilar_index] = (bias - (similarity_per_env > 0.5))[dissimilar_index]

        raw_bonus.append(raw_flat_eco_bonus)
        bonus.append(flat_eco_bonus)

    # 3. Aggregate All Bonus Calculated
    bonus = np.reshape(bonus, -1)
    raw_bonus = np.reshape(raw_bonus, -1)
    return bonus, raw_bonus

def main(argv):
  pass

if __name__ == '__main__':
  app.run(main)
