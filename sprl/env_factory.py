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

"""Factories to create DMLab env with episodic curiosity rewards."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import functools
import os
import gin
from absl import flags
from sprl import constants, curiosity_env_wrapper, r_network, r_network_training
from sprl.constants import Const
from sprl.environments import dmlab_utils
from third_party.baselines import logger
from third_party.baselines.bench import Monitor
from third_party.baselines.common.vec_env import subproc_vec_env
from third_party.baselines.common.vec_env import threaded_vec_env
from third_party.keras_resnet import models

DEFAULT_VEC_ENV_CLASS_NAME = 'SubprocVecEnv'


flags.DEFINE_enum('vec_env_class',
                  DEFAULT_VEC_ENV_CLASS_NAME,
                  ['SubprocVecEnv', 'ThreadedVecEnv'],
                  'Vec env class to use. ')

FLAGS = flags.FLAGS


def get_action_set(action_set_name):
  """Returns action sets by name."""
  return {
      '': dmlab_utils.DEFAULT_ACTION_SET,
      'small': dmlab_utils.ACTION_SET_SMALL,
      'nofire': dmlab_utils.DEFAULT_ACTION_SET_WITHOUT_FIRE,
      'withidle': dmlab_utils.ACTION_SET_WITH_IDLE,
      'defaultwithidle': dmlab_utils.DEFAULT_ACTION_SET_WITH_IDLE,
      'smallwithback': dmlab_utils.ACTION_SET_SMALL_WITH_BACK,
  }[action_set_name]


@gin.configurable
def create_single_env(env_name, seed, dmlab_homepath, use_monitor,
                      split='train', vizdoom_maze=False, action_set='',
                      respawn=True, fixed_maze=False, maze_size=None,
                      room_count=None, episode_length_seconds=None,
                      min_goal_distance=None, run_oracle_before_monitor=False,
                      ):
  """Creates a single instance of DMLab env, with training mixer seed.

  Args:
    env_name: Name of the DMLab environment.
    seed: seed passed to DMLab. Must be != 0.
    dmlab_homepath: Path to DMLab MPM. Required when running on borg.
    use_monitor: Boolean to add a Monitor wrapper.
    split: One of {"train", "valid", "test"}.
    vizdoom_maze: Whether a geometry of a maze should correspond to the one used
      by Pathak in his curiosity paper in Vizdoom environment.
    action_set: One of {'small', 'nofire', ''}. Which action set to use.
    respawn: If disabled respawns are not allowed
    fixed_maze: Boolean to use predefined maze configuration.
    maze_size: If not None sets particular height/width for mazes to be
      generated.
    room_count: If not None sets the number of rooms for mazes to be generated.
    episode_length_seconds: If not None overrides the episode duration.
    min_goal_distance: If not None ensures that there's at least this distance
      between the starting and target location (for
      explore_goal_locations_large level).
    run_oracle_before_monitor: Whether to run OracleRewardWrapper before the
      Monitor.

  Returns:
    Gym compatible DMLab env.

  Raises:
    ValueError: when the split is invalid.
  """
  main_observation = 'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE'
  level = constants.Const.find_level(env_name)
  env_settings = dmlab_utils.create_env_settings(
      level.dmlab_level_name,
      homepath=dmlab_homepath,
      width=Const.OBSERVATION_WIDTH,
      height=Const.OBSERVATION_HEIGHT,
      seed=seed,
      main_observation=main_observation)
  env_settings.update(level.extra_env_settings)

  if maze_size:
    env_settings['mazeHeight'] = maze_size
    env_settings['mazeWidth'] = maze_size
  if min_goal_distance:
    env_settings['minGoalDistance'] = min_goal_distance
  if room_count:
    env_settings['roomCount'] = room_count
  if episode_length_seconds:
    env_settings['episodeLengthSeconds'] = episode_length_seconds

  if split == 'train':
    mixer_seed = Const.MIXER_SEEDS[constants.SplitType.POLICY_TRAINING]
  elif split == 'valid':
    mixer_seed = Const.MIXER_SEEDS[constants.SplitType.VALIDATION]
  elif split == 'test':
    mixer_seed = Const.MIXER_SEEDS[constants.SplitType.TEST]
  else:
    raise ValueError('Invalid split: {}'.format(split))

  env_settings.update(mixerSeed=mixer_seed)

  if vizdoom_maze:
    env_settings['episodeLengthSeconds'] = 60
    env_settings['overrideEntityLayer'] = """*******************
*****   *   ***   *
*****             *
*****   *   ***   *
****** *** ***** **
*   *   *   ***   *
*P          ***   *
*   *   *   ***   *
****** ********* **
****** *********G**
*****   ***********
*****   ***********
*****   ***********
****** ************
****** ************
******   **********
*******************"""

  if fixed_maze:
    env_settings['overrideEntityLayer'] = """
*****************
*       *PPG    *
* *** * *PPP*** *
* *GPP* *GGG PGP*
* *GPG* * ***PGP*
* *PGP*   ***PGG*
* *********** * *
*     *GPG*GGP  *
* *** *PPG*PGG* *
*PGP* *GPP PPP* *
*PPP* * *** *** *
*GGG*     *GPP* *
*** ***** *GGG* *
*GPG PPG   PPP* *
*PGP*GGP* ***** *
*GPP*GPP*       *
*****************"""

  # Gym compatible environment.
  env = dmlab_utils.DMLabWrapper(
      'dmlab',
      env_settings,
      action_set=get_action_set(action_set),
      main_observation=main_observation,)
  if run_oracle_before_monitor:
    env = dmlab_utils.OracleRewardWrapper(env)

  if vizdoom_maze or not respawn:
    env = dmlab_utils.EndEpisodeOnRespawn(env)

  if use_monitor:
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(seed)))
  return env


@gin.configurable
def _create_r_net_and_trainer(
    obs_shape,
    num_envs,
    observation_history_size,
    tolerance,
    neg_sample_adder,
    max_action_distance,
    num_epochs,
    train_size,
    buffer_size,
    r_network_weights_store_path='',
    random_state_predictor=False,
    is_env=None,
    ):
  # 0. Set action-conditioning
  r_net_class = r_network.RNetwork
  r_net_trainer_class = r_network_training.RNetworkTrainer

  # 1. Build R-net
  r_net = r_net_class(obs_shape, random_state_predictor=random_state_predictor)

  # 2. If online, define R-net trainer
  r_network_trainer = r_net_trainer_class(
    r_net._r_network,  # pylint: disable=protected-access
    # Input
    num_envs=num_envs,
    observation_history_size=observation_history_size,
    # R-net param
    tolerance=tolerance,
    neg_sample_adder=neg_sample_adder,
    max_action_distance=max_action_distance,
    # Learning param
    num_epochs=num_epochs,
    train_size=train_size,
    buffer_size=buffer_size,
    # Etc
    checkpoint_dir=r_network_weights_store_path,
    is_env=is_env)
  return r_net, r_network_trainer

@gin.configurable
def create_environments(env_name,
                        num_envs,
                        dmlab_homepath = '',
                        r_network_weights_store_path = None,
                        # Don't change above
                        action_set = '',
                        base_seed = 123,
                        environment_engine = 'dmlab',
                        max_action_distance = 0,
                        policy_architecture = None,
                        tolerance = 0.,
                        gamma = 0.99,
                        random_state_predictor = False,
                        ):
  """Creates a environments with R-network-based curiosity reward.
  Use curiosityenvwrapper to give curiosity bonus! only need to modify this part
  online r network training flag is here
  curiosityenvwrapper addes rnetworktrainer as observer
  => observer.on_new_observation() at each step and calculate reward bonus in curiosityenvwrapper
  1. RNetTrainer.on_new_observation() adds observation in fifo memory
     Then it trains RNetTrainer by RNetTrainer.train()
     Already have validation in .train() function of trainer
        - get validation accuracy of r_network for log
     * max_action_distance of RNet is decided inside _prepare_data function
     and it is different from FLAGS.max_action_distance...
        - compare when use max action distance in FLAGS(train_r.py) and _prepare_data respectively
          - train_r.py is only used in offline version and for pretraining
            - in launcher_script.py, get_train_r_command in run_r_net_training in run_training function
          - _prepare_data function is only used in online version of RNet training
          - v2 is used for _prepare_data function

  Args:
    env_name: Name of the DMLab environment.
    num_envs: Number of parallel environment to spawn.
    dmlab_homepath: Path to the DMLab MPM. Required when running on borg.
    action_set: One of {'small', 'nofire', ''}. Which action set to use.
    base_seed: Each environment will use base_seed+env_index as seed.
    environment_engine: 'dmlab'.
    max_action_distance_list: List of max_action_distance for each reachability
      network. Define how many reachability network to use and which max_action
      _distance to use for each of them.

  Returns:
    Wrapped environment with curiosity.
  """
  # Environments without intrinsic exploration rewards.
  # pylint: disable=g-long-lambda
  create_single_dmlab_env = functools.partial(create_single_env,
                                              dmlab_homepath=dmlab_homepath,
                                              action_set=action_set)

  is_env = {'dmlab':False}
  if environment_engine == 'dmlab':
    create_env_fn = create_single_dmlab_env
    is_env['dmlab'] = True
    obs_shape = Const.OBSERVATION_SHAPE
    target_image_shape = [84, 84, 3]
  else:
    raise ValueError('Unknown env engine {}'.format(environment_engine))

  # WARNING: python processes are not really compatible with other google3 code,
  # which can lead to deadlock. See go/g3process. This is why you can use
  # ThreadedVecEnv.
  VecEnvClass = (subproc_vec_env.SubprocVecEnv
                 if FLAGS.vec_env_class == 'SubprocVecEnv'
                 else threaded_vec_env.ThreadedVecEnv)
  vec_env = VecEnvClass([
      (lambda _i=i: create_env_fn(env_name, base_seed + _i, use_monitor=True,
                                  split='train'))
      for i in range(num_envs)
  ])

  num_test_envs = 1
  test_env = VecEnvClass([
      (lambda _i=i: create_env_fn(env_name, base_seed + _i, use_monitor=False,
                                  split='test'))
      for i in range(num_test_envs)
  ])# pylint: enable=g-long-lambda

  if max_action_distance > 0:  # If using R-net
    # 1. Create R-net & R-net-trainer
    r_net, r_network_trainer = _create_r_net_and_trainer(
        obs_shape=obs_shape,
        num_envs=num_envs,
        tolerance=tolerance,
        max_action_distance=max_action_distance,
        r_network_weights_store_path=r_network_weights_store_path,
        random_state_predictor=random_state_predictor,
        is_env=is_env,
    )
    observation_embedding_fn = r_net.embed_observation
    observation_compare_fn = r_net.embedding_similarity

    embedding_size = [models.EMBEDDING_DIM]
  else:  # No R-net (ppo | ICM)
    r_net = None
    r_network_trainer = None
    observation_embedding_fn = None
    observation_compare_fn = None
    embedding_size = None

  # 3. Build environment wrapper
  env_wrapper = curiosity_env_wrapper.CuriosityEnvWrapper(
      vec_env, observation_embedding_fn, observation_compare_fn,
      target_image_shape,
      r_network_trainer=r_network_trainer,
      r_net=r_net,
      embedding_size=embedding_size,
      max_action_distance=max_action_distance, tolerance=tolerance,
      policy_architecture=policy_architecture, gamma=gamma,
      r_network_weights_store_path=r_network_weights_store_path,
      is_env=is_env,)
  test_env_wrapper = curiosity_env_wrapper.CuriosityEnvWrapper(
      test_env, observation_embedding_fn, observation_compare_fn,
      target_image_shape,
      r_network_trainer=r_network_trainer,
      r_net=r_net,
      embedding_size = embedding_size,
      max_action_distance=max_action_distance, tolerance=tolerance,
      policy_architecture=policy_architecture, gamma=gamma,
      r_network_weights_store_path=r_network_weights_store_path,
      is_env=is_env,)
  return env_wrapper, test_env_wrapper
