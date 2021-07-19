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

"""Main file for training policies.

Many hyperparameters need to be passed through gin flags.
Consider using scripts/launcher_script.py to invoke train_policy with the
right hyperparameters and flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import tensorflow as tf

from absl import flags
from sprl import env_factory, eval_policy
from sprl.utils import Summary_logger
from third_party.baselines import logger
from third_party.baselines.ppo2 import policies
from third_party.baselines.ppo2 import ppo2
from third_party.baselines.common import set_global_seeds

flags.DEFINE_string('workdir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('seed', 1,
                    'Global random seed')
flags.DEFINE_string('env_name', 'goal_small', 'What environment to run')
flags.DEFINE_enum('policy_architecture', 'cnn',
                  ['cnn'],
                  'which architecture to use for policy')
flags.DEFINE_string('r_checkpoint', '', 'Location of the R-network checkpoint')
flags.DEFINE_integer('num_env', 12, 'Number of environment copies to run in '
                     'subprocesses.')
flags.DEFINE_string('dmlab_homepath', '', '')
flags.DEFINE_integer('num_timesteps', 10000000, 'Number of frames to run '
                     'training for.')
flags.DEFINE_integer('early_stop_timesteps', -1, 'Number of frames to stop '
                     'training after.')
flags.DEFINE_string('action_set', '',
                    '(small|nofire|) - which action set to use')
flags.DEFINE_bool('random_state_predictor', False,
                  'Whether to use random state predictor for Pathak\'s '
                  'curiosity')
flags.DEFINE_float('early_rnet_train_freq',
                  2.0,
                  'rnet training interval for early phase'
                  )
flags.DEFINE_float('mid_rnet_train_freq',
                  6.0,
                  'rnet training interval for mid phase'
                  )
flags.DEFINE_float('last_rnet_train_freq',
                  12.0,
                  'rnet training interval for last phase'
                  )
flags.DEFINE_integer('early_rnet_steps',
                  1000000,
                  'early rnet training steps'
                  )

# pylint: disable=g-inconsistent-quotes
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
# pylint: enable=g-inconsistent-quotes

FLAGS = flags.FLAGS

def get_environment(env_name,policy_architecture,gamma,random_state_predictor):
  for prefix in ['dmlab:']:
    if env_name.startswith(prefix):
      level_name = env_name[len(prefix):]
      return env_factory.create_environments(
          level_name,
          FLAGS.num_env,
          dmlab_homepath=FLAGS.dmlab_homepath,
          r_network_weights_store_path=FLAGS.workdir,
          action_set=FLAGS.action_set,
          environment_engine=prefix[:-1],
          policy_architecture=policy_architecture,
          gamma=gamma,
          random_state_predictor=random_state_predictor,)
  raise ValueError('Unknown environment: {}'.format(env_name))

@gin.configurable
def train(workdir, env_name, num_timesteps,
          nsteps=256,
          nminibatches=4,
          noptepochs=4,
          learning_rate=2.5e-4,
          max_grad_norm=0.5,
          ent_coef=0.01,
          gamma=0.99,
          use_curiosity=False,
          curiosity_strength=0.55,
          forward_inverse_ratio=0.96,
          curiosity_loss_strength=64):
  """Runs PPO training. prepares environment and run ppo2.learn

  Args:
    workdir: where to store experiment results/logs
    env_name: the name of the environment to run
    num_timesteps: for how many timesteps to run training
    nsteps: Number of consecutive environment steps to use during training.
    nminibatches: Minibatch size.
    noptepochs: Number of optimization epochs.
    learning_rate: Initial learning rate.
    ent_coef: Entropy coefficient.
    use_curiosity => icm
  """
  logger_dir = workdir
  logger.configure(dir=logger_dir)
  tf_log_dir=os.path.join(workdir, "tf_logs")
  summary_logger = Summary_logger(tf_log_dir)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  env, test_env = get_environment(env_name,FLAGS.policy_architecture,gamma,FLAGS.random_state_predictor)

  policy_evaluator_on_test = eval_policy.PolicyEvaluator(
      test_env,
      workdir=FLAGS.workdir,)

  cloud_sync_callback = lambda: None

  def evaluate_valid_test(model_step_fn):
    policy_evaluator_on_test.evaluate(model_step_fn)

  with sess.as_default():
    policy = {'cnn': policies.CnnPolicy,
              }[FLAGS.policy_architecture]

    ppo2.learn(policy, env=env, nsteps=nsteps, nminibatches=nminibatches,
               lam=0.95, gamma=gamma, noptepochs=noptepochs, log_interval_per_update=1,
               max_grad_norm=max_grad_norm,
               ent_coef=lambda f: ent_coef,
               lr=lambda f: f * learning_rate,
               cliprange=lambda f: f * 0.1,
               total_timesteps=num_timesteps+FLAGS.num_env*nsteps,
               eval_callback=evaluate_valid_test,
               cloud_sync_callback=cloud_sync_callback,
               save_interval=2000, workdir=workdir, summary_logger=summary_logger,
               use_curiosity=use_curiosity,
               curiosity_strength=curiosity_strength,
               forward_inverse_ratio=forward_inverse_ratio,
               curiosity_loss_strength=curiosity_loss_strength,
               random_state_predictor=FLAGS.random_state_predictor,
               early_stop_timesteps=FLAGS.early_stop_timesteps,
               early_rnet_train_freq=FLAGS.early_rnet_train_freq,
               mid_rnet_train_freq=FLAGS.mid_rnet_train_freq,
               last_rnet_train_freq=FLAGS.last_rnet_train_freq,
               early_rnet_steps=FLAGS.early_rnet_steps)
    cloud_sync_callback()
  test_env.close()

def main(_): # called from launcher_script.py ; run ppo2.learn
  set_global_seeds(FLAGS.seed)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_files,
                                      FLAGS.gin_bindings)

  train(FLAGS.workdir, env_name=FLAGS.env_name,
        num_timesteps=FLAGS.num_timesteps)


if __name__ == '__main__':
  tf.app.run()
