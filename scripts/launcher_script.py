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

"""
Script that launches policy training with the right hyperparameters.
All specified runs are launched in parallel as subprocesses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import datetime
import subprocess

from absl import app
from absl import flags

from sprl import constants
from sprl.utils import dump_flags_to_file, dump_dicts_to_file
from third_party.baselines.common import set_global_seeds

import six
import tensorflow as tf

SHORT_NAMES = {
  'policy_lr': ('lr', 'g'),
  'policy_ent_coef': ('ent', 'g'),
  'policy_archi': ('model', 's'),
  'num_env': ('Nenv', 'd'),
  'max_action_distance': ('K', 'd'),
  'tolerance': ('tol', 'd'),
  'neg_sample_adder': ('neg_th', 'd'),
  'l2_norm': ('RL2', 'g'),
  'scale_shortest_bonus': ('sprl_scale', 'f'),
  'bonus_bias': ('bias', 'f'),
  'tol_num': ('tol_num', 'd'),
  'bonus_thres': ('bthres', 'f2'),
  'scale_oracle_reward': ('oracle', 'g'),
  'curiosity_strength': ('icm_str', 'g'),
  'forward_inverse_ratio': ('fwd_ratio', 'f'),
  'curiosity_loss_strength': ('icm_loss_str', 'g'),
  'scale_eco_bonus': ('eco_scale', 'g'),
  'epmem_size': ('epmem_size', 'd'),
}
DEFAULT = ['policy_archi', 'policy_lr', 'policy_ent_coef', 'num_env']
R_NET_PARAMS = ['max_action_distance', 'tolerance']
R_NET_TRAIN_PARAMS = ['neg_sample_adder', 'l2_norm']
SPRL_PARAMS = ['scale_shortest_bonus', 'bonus_bias', 'tol_num', 'bonus_thres']
ICM_PARAMS = ['curiosity_strength', 'forward_inverse_ratio', 'curiosity_loss_strength']
ECO_PARAMS = ['scale_eco_bonus', 'epmem_size', 'bonus_bias', 'bonus_thres']
ORACLE_PARAMS = ['scale_oracle_reward']
PARAMS = set(DEFAULT+R_NET_PARAMS+R_NET_TRAIN_PARAMS+SPRL_PARAMS+ICM_PARAMS+ECO_PARAMS+ORACLE_PARAMS)
if set(SHORT_NAMES.keys()) != PARAMS:
  short_names = set(SHORT_NAMES.keys())
  print('missing in PARAMS: ', short_names - PARAMS)
  print('missing in SHORT_NAMES: ', PARAMS - short_names)
  assert False, 'Error. PARAMS != SHORT_NAMES'

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', 'workdir',
                    'Directory where all experiment results will be stored')
flags.mark_flag_as_required('workdir')

flags.DEFINE_enum(
    'method', 'eco',
    ['ppo', 'eco', 'oracle', 'sprl', 'icm'],
    'Solving method to use.')

DMLAB_SCENARIOS = ['sparse', 'dense_explore', 'goal_small']

REQUIRE_RNET = ['sprl', 'eco']

flags.DEFINE_enum('scenario', 'goal_small',
                  DMLAB_SCENARIOS,
                  'Scenario to launch.')

flags.DEFINE_integer('num_timesteps', 20000000,
                     'Number of training timesteps to run.')
flags.DEFINE_integer('early_stop_timesteps', -1,
                     'Number of training timesteps to stop.')
flags.DEFINE_integer('num_env', 6,
                     'Number of envs to run in parallel for training the '
                     'policy.')

flags.DEFINE_integer('max_action_distance',
                  0,
                  'max action distance(positive sample criteria) for '
                  'reachability network.'
                  )
flags.DEFINE_integer('tol_num',
                  200,
                  'number of tolerance to use'
                  )
flags.DEFINE_enum('policy_archi', 'cnn',
                  ['cnn'],
                  'which architecture to use for policy')
flags.DEFINE_float('bonus_bias',
                  0.0,
                  'bias being added to sprl_bonus or eco_bonus'
                  )
flags.DEFINE_boolean('use_various_tol',
                  False,
                  'flag for using various tol'
                  )
flags.DEFINE_integer('hidden_layer_size',
                  256,
                  'hidden layer size for inverse dynamics feature model'
                  )
flags.DEFINE_integer('tolerance',
                  0,
                  'which order of mdp are we going to use as modified mdp'
                  'max_action_distance + tolerance = temporal difference(steps)'
                  'between (state1, state2) pair for calculating bonus & stacking states'
                  )
flags.DEFINE_float('neg_sample_adder',
                  25,
                  'gamma in negative sample criteria gamma + k'
                  )
flags.DEFINE_float('scale_shortest_bonus',
                  0.0, #0.03017241379310345
                  'coefficient for sprl bonus'
                  )
flags.DEFINE_float('scale_eco_bonus',
                  0.0,
                  'coefficient for eco bonus'
                  )
flags.DEFINE_float('scale_oracle_reward',
                  0.05246913580246913,
                  'coefficient for oracle reward'
                  )
flags.DEFINE_float('scale_task_reward',
                  1.0,
                  'coefficient for external reward'
                  )
flags.DEFINE_integer('rnet_buffer_size',
                  60000,
                  'maximum number of observation that can be in the buffer'
                  'for rnet training'
                  )
flags.DEFINE_float('rnet_train_interval',
                  2,
                  'Train frequency (in # ppo updates)'
                  )
flags.DEFINE_integer('early_rnet_steps',
                  1000000,
                  'Rnet training steps for early phase'
                  )
flags.DEFINE_integer('rnet_num_epochs',
                  1,
                  'number of epochs for reachability network training'
                  )
flags.DEFINE_integer('mdpadder',
                  0,
                  'compare s_t with s_t-k-mdpadder not s_t-k'
                  )
flags.DEFINE_integer('rnet_train_size',
                  30000,
                  'The number of data for reachability network training'
                  )
flags.DEFINE_float('policy_lr',
                  2.5e-4,
                  'learning rate for policy'
                  )
flags.DEFINE_float('rnet_lr',
                  3.0e-4,
                  'learning rate for rnet'
                  )
flags.DEFINE_integer('neg_pos_ratio',
                  1,
                  'how many negative samples to train when 1 positive sample is being trained in rnet training'
                  )
flags.DEFINE_float('neg_upper_ratio',
                  0.1,
                  'neg_sample_threshold ~ neg_sample_threshold*upper_ratio is used for neg sample sampling in rnet training'
                  )
flags.DEFINE_float('bonus_thres',
                  90,
                  'percentile to use for bonus'
                  )
flags.DEFINE_float('l2_norm',
                  0.0001,
                  'l2 norm for top network of rnet / Note: not for resnet which is an embedding network for rnet'
                  )
flags.DEFINE_float('max_grad_norm',
                  0.5,
                  'gradient clipping'
                  )
flags.DEFINE_float('policy_ent_coef',
                  0.002053525026457146,
                  'coefficient for entropy of policy'
                  )
flags.DEFINE_boolean('use_level_cache',
                    True,
                    'input True to use level cache for caching')
flags.DEFINE_integer('seed',
                  123,
                  'seed for tf, random, numpy, dmlab'
                  )
# === ICM ===
flags.DEFINE_float('curiosity_strength', 0.1,
                   'Strength of the intrinsic reward in Pathak\'s algorithm.')
flags.DEFINE_float('forward_inverse_ratio', 1.0,
                   'Strength of the intrinsic reward in Pathak\'s algorithm.')
flags.DEFINE_float('curiosity_loss_strength', 2.0,
                     'coefficient for loss in icm')
# === ECO ===
flags.DEFINE_integer('epmem_size',
                  200,
                  'size of episodic memory'
                  )
PYTHON_BINARY = 'python'

def logged_check_call(command):
  """Logs the command and calls it."""
  subprocess.check_call(command)

def flatten_list(to_flatten):
  # pylint: disable=g-complex-comprehension
  return [item for sublist in to_flatten for item in sublist]

def quote_gin_value(v):
  if isinstance(v, six.string_types):
    return '"{}"'.format(v)
  return v

def assemble_command(base_command, params):
  """Builds a command line to launch training.

  Args:
    base_command: list(str), command prefix.
    params: dict of param -> value. Parameters prefixed by '_gin.' are
      considered gin parameters.

  Returns:
    List of strings, the components of the command line to run.
  """
  gin_params = {param_name: param_value
                for param_name, param_value in params.items()
                if param_name.startswith('_gin.')}
  params = {param_name: param_value
            for param_name, param_value in params.items()
            if not param_name.startswith('_gin.')}
  return (base_command +
          ['--{}={}'.format(param, v)
           for param, v in params.items()] +
          flatten_list([['--gin_bindings',
                         '{}={}'.format(gin_param[len('_gin.'):],
                                        quote_gin_value(v))]
                        for gin_param, v in gin_params.items()]))

def get_common_params(scenario):
  common_params = {
    'policy_architecture': FLAGS.policy_archi,
    '_gin.train.ent_coef': FLAGS.policy_ent_coef,
    '_gin.train.learning_rate': FLAGS.policy_lr,
    '_gin.train.max_grad_norm': FLAGS.max_grad_norm,
    '_gin.DMLabWrapper.use_level_cache': FLAGS.use_level_cache,
    '_gin.Model.hidden_layer_size': FLAGS.hidden_layer_size,
    'action_set': '',
    '_gin.CuriosityEnvWrapper.scale_task_reward': FLAGS.scale_task_reward,
    }
  return common_params

def get_rnet_params():
  r_net_params = {
    '_gin.create_environments.max_action_distance' : FLAGS.max_action_distance,
    #
    '_gin.build_siamese_resnet_18.hidden_layer_size': 288,
    '_gin.build_siamese_resnet_18.l2_norm': FLAGS.l2_norm,
    #
    '_gin._create_r_net_and_trainer.num_epochs': FLAGS.rnet_num_epochs,
    '_gin._create_r_net_and_trainer.train_size': FLAGS.rnet_train_size,
    '_gin._create_r_net_and_trainer.observation_history_size': FLAGS.rnet_buffer_size,
    '_gin._create_r_net_and_trainer.buffer_size': FLAGS.rnet_buffer_size,
    '_gin._create_r_net_and_trainer.neg_sample_adder': FLAGS.neg_sample_adder,
    '_gin.generate_negative_example.neg_sample_upper_ratio': FLAGS.neg_upper_ratio,
    '_gin.sample_triplet_data.neg_pos_ratio': FLAGS.neg_pos_ratio,
  }

  r_net_params.update({
    '_gin.RNetwork.learning_rate': FLAGS.rnet_lr,
  })
  return r_net_params

def get_oracle_params(scenario):
  # Returns the param for the 'oracle' method.
  params = get_common_params(scenario)
  params.update({
    '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
    '_gin.CuriosityEnvWrapper.scale_oracle_reward': FLAGS.scale_oracle_reward,
    '_gin.OracleExplorationReward.reward_grid_size': 30,
  })
  return params

def get_icm_params(scenario):
  # Returns the param for the 'icm' method.
  params = get_common_params(scenario)
  params.update({
    '_gin.train.use_curiosity': True,
    '_gin.CuriosityEnvWrapper.exploration_reward': 'none',
    # Below parameters => using default values in the paper
    '_gin.BonusModel.bonus_type': ('icm'),
    '_gin.train.forward_inverse_ratio': FLAGS.forward_inverse_ratio,
    '_gin.train.curiosity_loss_strength': FLAGS.curiosity_loss_strength,
    '_gin.train.curiosity_strength': FLAGS.curiosity_strength,
  })
  return params

def get_ppo_params(scenario):
  # Returns the param for the 'ppo' method.
  params = get_common_params(scenario)
  params.update({
    '_gin.BonusModel.scale_shortest_bonus': 0,
    '_gin.BonusModel.scale_eco_bonus': 0,
    '_gin.CuriosityEnvWrapper.exploration_reward': 'none',
    '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
  })
  return params

def get_eco_params(scenario):
  """Returns the param for the 'sprl' method."""
  assert scenario in DMLAB_SCENARIOS, (
      'Non-DMLab scenarios not supported as of today by PPO+SPRL method')
  assert FLAGS.method in ['eco']
  params = get_common_params(scenario)
  rnet_params = get_rnet_params()
  params.update(rnet_params)
  params.update({
  # ECO
    '_gin.CuriosityEnvWrapper.exploration_reward': FLAGS.method,
    '_gin.BonusModel.bonus_type': 'eco',
    '_gin.BonusModel.bonus_bias': FLAGS.bonus_bias,
    '_gin.BonusModel.epmem_size': FLAGS.epmem_size,
    '_gin.BonusModel.bonus_thres': FLAGS.bonus_thres,
    '_gin.BonusModel.scale_eco_bonus': FLAGS.scale_eco_bonus,
  })
  return params

def get_sprl_params(scenario):
  """Returns the param for the 'sprl' method."""
  assert scenario in DMLAB_SCENARIOS, (
      'Non-DMLab scenarios not supported as of today by PPO+SPRL method')
  assert FLAGS.method in ['sprl']
  params = get_common_params(scenario)
  rnet_params = get_rnet_params()
  params.update(rnet_params)
  params.update({
  # Sprl
    '_gin.create_environments.tolerance': FLAGS.tolerance,
    '_gin.CuriosityEnvWrapper.exploration_reward': FLAGS.method,
    '_gin.CuriosityEnvWrapper.max_total_env_step': FLAGS.num_timesteps,
    '_gin.BonusModel.scale_shortest_bonus': FLAGS.scale_shortest_bonus,
    '_gin.BonusModel.bonus_type': ('sprl'),
    '_gin.BonusModel.bonus_bias': FLAGS.bonus_bias,
    '_gin.BonusModel.use_various_tol': FLAGS.use_various_tol,
    '_gin.BonusModel.bonus_thres': FLAGS.bonus_thres,
    '_gin.BonusModel.mdpadder': FLAGS.mdpadder,
    '_gin.BonusModel.tol_num': FLAGS.tol_num,
  })
  return params

################################################################

def run_training(workdir):
  """Runs training according to flags."""
  r_net_workdir = None

  if FLAGS.method == 'sprl':
    policy_training_params = get_sprl_params(FLAGS.scenario)
    policy_training_params.update({
      '_gin.CuriosityEnvWrapper.exploration_reward': FLAGS.method,
      })
  elif FLAGS.method == 'icm':
    policy_training_params = get_icm_params(FLAGS.scenario)
  elif FLAGS.method == 'ppo':
    policy_training_params = get_ppo_params(FLAGS.scenario)
  elif FLAGS.method == 'eco':
    policy_training_params = get_eco_params(FLAGS.scenario)
  elif FLAGS.method == 'oracle':
    policy_training_params = get_oracle_params(FLAGS.scenario)
    policy_training_params.update({
      '_gin.BonusModel.scale_shortest_bonus': 0,})
  else:
    raise NotImplementedError(
        'method {} is not implemented.'.format(FLAGS.method))

  if FLAGS.scenario in DMLAB_SCENARIOS:
    env_name = ('dmlab:' + constants.Const.find_level_by_scenario(
        FLAGS.scenario).fully_qualified_name)
    assert 'cnn' in FLAGS.policy_archi
  else:
    assert FLAGS.scenario
    raise NotImplementedError

  policy_training_params.update({
      'seed': FLAGS.seed,
      'workdir': workdir,
      'num_env': str(FLAGS.num_env),
      'env_name': env_name,
      'num_timesteps': FLAGS.num_timesteps,
      'early_stop_timesteps': FLAGS.early_stop_timesteps,
      '_gin.create_environments.base_seed': FLAGS.seed,
      '_gin.CuriosityEnvWrapper.scale_task_reward': FLAGS.scale_task_reward,
      })

  rnet_train_dict = {'early_rnet_train_freq': FLAGS.rnet_train_interval,
      'mid_rnet_train_freq': FLAGS.rnet_train_interval*3,
      'last_rnet_train_freq': FLAGS.rnet_train_interval*6,
      'early_rnet_steps': FLAGS.early_rnet_steps}
  policy_training_params.update(rnet_train_dict)

  tf.io.gfile.makedirs(workdir)
  dump_flags_to_file(os.path.join(workdir, 'flags.txt'))
  dump_dicts_to_file(os.path.join(workdir, 'flags.txt'), rnet_train_dict)
  base_command = [PYTHON_BINARY, '-m', 'sprl.train_policy']
  assemble_command(base_command, policy_training_params)
  logged_check_call(assemble_command(
      base_command, policy_training_params))

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unexpected command line arguments.')

  if not tf.io.gfile.exists(FLAGS.workdir):
    tf.io.gfile.makedirs(FLAGS.workdir)
  set_global_seeds(FLAGS.seed)

  RELEVENT_PARAMS_BY_METHOD = {
    'ppo':  DEFAULT,
    'eco':  DEFAULT + R_NET_PARAMS + R_NET_TRAIN_PARAMS + ECO_PARAMS,
    'sprl': DEFAULT + R_NET_PARAMS + R_NET_TRAIN_PARAMS + SPRL_PARAMS,
    'oracle': DEFAULT + ORACLE_PARAMS,
    'icm': DEFAULT + ICM_PARAMS,
  }

  ### ==== Assertions to avoid mistake in parameter setting ==== ###
  if FLAGS.method == 'ppo':
    assert FLAGS.max_action_distance == 0 and FLAGS.tolerance == 0

  # Set directory name
  task_str = FLAGS.scenario
  method_str = FLAGS.method
  hparam_str = ""
  relevant_param_list = []
  for method, params in RELEVENT_PARAMS_BY_METHOD.items():
    if method in method_str:
        relevant_param_list += params
  refined_param_list = [] # Remove redundancy while preserving order
  [refined_param_list.append(x) for x in relevant_param_list if x not in refined_param_list]
  del relevant_param_list

  for arg in refined_param_list:
    if arg in SHORT_NAMES:
      short_arg, dtype = SHORT_NAMES[arg]
      value = FLAGS.flag_values_dict()[arg]
      prev_len = len(hparam_str)
      if dtype == 's':  # string
        hparam_str += '%s=%s'%(short_arg, value)
      elif dtype == 'd':
        hparam_str += '%s=%d'%(short_arg, value)
      elif dtype == 'f':
        hparam_str += '%s=%g'%(short_arg, value)
      elif dtype == 'f1':
        hparam_str += '%s=%.1f'%(short_arg, value)
      elif dtype == 'f2':
        hparam_str += '%s=%.2f'%(short_arg, value)
      elif dtype == 'b':
        if value:
          hparam_str += '%s=1'%(short_arg)
        if not value:
          hparam_str += '%s=0'%(short_arg)
      elif dtype == 'g':
        hparam_str += '%s=%g'%(short_arg, value)
      elif dtype == 'e':
        hparam_str += '%s=%.2e'%(short_arg, value)
      elif dtype == 'ls': # list of string
        hparam_str += '%s='%(short_arg)
        for val_ind in range(len(value)):
          hparam_str += '%s'%(value[val_ind])
          if val_ind != (len(value)-1):
            hparam_str += '_'
      #
      if arg != refined_param_list[-1]:
        hparam_str += '_'
  #
  workdir = os.path.join(os.path.expanduser(FLAGS.workdir),
                        task_str,
                        method_str + '__' + hparam_str,
                        'run_{}_{}'.format(FLAGS.seed, datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
                        )
  # Run training
  run_training(workdir)


if __name__ == '__main__':
  app.run(main)
