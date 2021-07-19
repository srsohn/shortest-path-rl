# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque, defaultdict

import os
import os.path as osp
import gin
import time
import math
import dill
import numpy as np
import tensorflow as tf

from PIL import Image
from third_party.baselines import logger
from third_party.baselines.common import tf_util as U
from third_party.baselines.common import explained_variance
from third_party.baselines.common.input import observation_input
from third_party.baselines.common.runners import AbstractEnvRunner
from third_party.baselines.ppo2 import pathak_utils
from third_party.baselines.ppo2 import policies
from sprl.bonus_model import BonusModel
from sprl.utils import _flatten, _unflatten, dump_pickle, load_pickle

@gin.configurable
class Model(object):
  def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train,
               nsteps, ent_coef, vf_coef, max_grad_norm,
               use_curiosity, curiosity_strength, forward_inverse_ratio,
               curiosity_loss_strength, random_state_predictor, pathak_multiplier,
               hidden_layer_size):
    sess = tf.get_default_session()
    act_model = policy(sess, ob_space, ac_space, nbatch_act, reuse=False)
    train_model = policy(sess, ob_space, ac_space, nbatch_train, reuse=True)
    self.nbatch_act = nbatch_act
    self.nbatch_train = nbatch_train
    self.action_dim = ac_space.n

    if use_curiosity:
      self.state_encoder_net = tf.make_template(
        'state_encoder_net', pathak_utils.universeHead,
        create_scope_now_=True,
        trainable=(not random_state_predictor),)
      self.icm_forward_net = tf.make_template(
          'icm_forward', pathak_utils.icm_forward_model,
          create_scope_now_=True, num_actions=ac_space.n,
          hidden_layer_size=hidden_layer_size, one_hot=(ac_space.dtype!='float32'))
      self.icm_forward_output = tf.make_template(
          'icm_forward_output', pathak_utils.icm_forward_output,
          create_scope_now_=True)
      self.icm_inverse_net = tf.make_template(
          'icm_inverse', pathak_utils.icm_inverse_model,
          create_scope_now_=True,
          hidden_layer_size=hidden_layer_size)
      self.icm_inverse_output = tf.make_template(
          'icm_inverse_output', pathak_utils.icm_inverse_output,
          create_scope_now_=True)
    else:
      self.state_encoder_net = None
      self.icm_forward_net = None
      self.icm_forward_output = None
      self.icm_inverse_net = None
      self.icm_inverse_output = None

    A = train_model.pdtype.sample_placeholder([None])
    ADV = tf.placeholder(tf.float32, [None])
    R = tf.placeholder(tf.float32, [None])
    OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
    OLDVPRED = tf.placeholder(tf.float32, [None])
    LR = tf.placeholder(tf.float32, [])
    CLIPRANGE = tf.placeholder(tf.float32, [])
    # When computing intrinsic reward a different batch size is used (number
    # of parallel environments), thus we need to define separate
    # placeholders for them.
    X_NEXT, _ = observation_input(ob_space, nbatch_train)
    X_INTRINSIC_NEXT, _ = observation_input(ob_space, None)
    X_INTRINSIC_CURRENT, _ = observation_input(ob_space, None)

    neglogpac = train_model.pd.neglogp(A)
    entropy = tf.reduce_mean(train_model.pd.entropy())

    vpred = train_model.vf
    vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED,
                                               - CLIPRANGE, CLIPRANGE)
    vf_losses1 = tf.square(vpred - R)
    vf_losses2 = tf.square(vpredclipped - R)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
    pg_losses = -ADV * ratio
    pg_losses2 = -ADV * tf.clip_by_value(ratio,
                                         1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
    clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0),
                                                     CLIPRANGE)))

    forward_loss, inverse_loss = self.compute_curiosity_loss(
        use_curiosity, train_model.X, A, X_NEXT,
        forward_inverse_ratio=forward_inverse_ratio,
        curiosity_loss_strength=curiosity_loss_strength)
    curiosity_loss = forward_loss + inverse_loss

    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + curiosity_loss

    if use_curiosity:
      encoded_time_step = self.state_encoder_net(X_INTRINSIC_CURRENT)
      encoded_next_time_step = self.state_encoder_net(X_INTRINSIC_NEXT)
      intrinsic_reward = self.curiosity_forward_model_loss(
          encoded_time_step, A, encoded_next_time_step)

    with tf.variable_scope('model'):
      params = tf.trainable_variables()

    grads = tf.gradients(loss * pathak_multiplier, params)
    if max_grad_norm is not None:
      grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
    _train = trainer.apply_gradients(grads)

    def getIntrinsicReward(curr, next_obs, actions):
      return sess.run(intrinsic_reward, {X_INTRINSIC_CURRENT: curr,
                                         X_INTRINSIC_NEXT: next_obs,
                                         A: actions})

    def train(lr, cliprange, obs, next_obs, returns, actions, values,
              neglogpacs):
      advs = returns - values
      advs = (advs - advs.mean()) / (advs.std() + 1e-8)
      # Inputs
      td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs,
                OLDVPRED: values, X_NEXT: next_obs}

      # Output
      summaries = [loss, pg_loss, vf_loss, entropy]
      if use_curiosity:
        summaries += [forward_loss, inverse_loss]
      return sess.run(summaries + [_train], td_map)[:-1]

    self.loss_names = ['loss', 'loss/policy', 'loss/value', 'policy_entropy']

    if use_curiosity:
      self.loss_names += ['loss/forward', 'loss/inverse']

    def save(save_path):
      ps = sess.run(params)
      with tf.gfile.Open(save_path, 'wb') as fh:
        fh.write(dill.dumps(ps))

    def load(load_path):
      with tf.gfile.Open(load_path, 'rb') as fh:
        val = fh.read()
        loaded_params = dill.loads(val)
      restores = []
      for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
      sess.run(restores)

    self.getIntrinsicReward = getIntrinsicReward
    self.train = train
    self.train_model = train_model
    self.act_model = act_model
    self.step = act_model.step
    self.eval_step = act_model.eval_step
    self.value = act_model.value
    self.save = save
    self.load = load
    tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101

  def curiosity_forward_model_loss(self, encoded_state, action,
                                   encoded_next_state):
    pred_next_state = self.icm_forward_output(encoded_state, self.icm_forward_net(encoded_state, action))
    forward_loss = 0.5 * tf.reduce_mean(
        tf.squared_difference(pred_next_state, encoded_next_state), axis=1)
    forward_loss = forward_loss * 288.0
    return forward_loss

  def curiosity_inverse_model_loss(self, encoded_states, actions,
                                   encoded_next_states):
    pred_action_logits = self.icm_inverse_output(self.icm_inverse_net(encoded_states,
                                              encoded_next_states), self.action_dim)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_action_logits, labels=actions), name='invloss')

  def compute_curiosity_loss(self, use_curiosity, time_steps, actions,
                             next_time_steps, forward_inverse_ratio,
                             curiosity_loss_strength):
    if use_curiosity:
      with tf.name_scope('curiosity_loss'):
        encoded_time_steps = self.state_encoder_net(time_steps)
        encoded_next_time_steps = self.state_encoder_net(next_time_steps)

        inverse_loss = self.curiosity_inverse_model_loss(
            encoded_time_steps, actions, encoded_next_time_steps)
        forward_loss = self.curiosity_forward_model_loss(
            encoded_time_steps, actions, encoded_next_time_steps)
        forward_loss = tf.reduce_mean(forward_loss)
    else:
      forward_loss = tf.constant(0.0, dtype=tf.float32,
                                         name='forward_loss')
      inverse_loss = tf.constant(0.0, dtype=tf.float32,
                                         name='inverse_loss')
    return curiosity_loss_strength * (forward_inverse_ratio * forward_loss), curiosity_loss_strength * (1 - forward_inverse_ratio) * inverse_loss

class Runner(AbstractEnvRunner):
  def __init__(self, env, model, bonus_model, rnet_trainer, rnet, nsteps, gamma, lam,
               log_dir='.', eval_callback=None):
    super(Runner, self).__init__(env=env, model=model, nsteps=nsteps)
    self._eval_callback = eval_callback

    self.lam = lam
    self.gamma = gamma
    self.nenvs = self.env.num_envs
    self.action_dim = self.env.action_space.n
    #
    self.bonus_model = bonus_model
    self.rnet_trainer = rnet_trainer
    self.rnet = rnet
    #
    self._collection_iteration = 0
    self.bonus_accum_dict = {
      'sprl': np.zeros(self.nenvs),
      'eco': np.zeros(self.nenvs),
      'icm': np.zeros(self.nenvs),
      'bonus': np.zeros(self.nenvs),
    }
    self.infos = env.get_initial_infos()
    self._ep_return = np.zeros(self.nenvs)
    self._ep_length = np.zeros(self.nenvs)
    self.log_dir = log_dir
    self.rewards = 0

  def _update_accumulator(self, mb_dones, current_dict, epinfos, accum_dict):
    for epinfo in epinfos:
      for key in accum_dict:
        epinfo[key] = 0.

    count = 0
    # mb_dones => [256, num_envs]
    for idx, batch_done in enumerate(mb_dones): # idx = 0~255
      # batch_done => [num_envs]
      for key, batch_current in current_dict.items():
        # batch_current : [256 x num_envs]
        current_vec = batch_current[idx] # [num_envs]
        if key in accum_dict:
          accum_dict[key] += current_vec

      for env_ind, done in enumerate(batch_done):  # for each env: 0~11
        if done:
          for key in accum_dict:
            epinfos[count][key] = accum_dict[key][env_ind].copy()
            accum_dict[key][env_ind] = 0.
          count += 1

    return epinfos

  @gin.configurable
  def run(self):
    if self._eval_callback:
      self._eval_callback(self.model.eval_step)

    self._collection_iteration += 1

    mb_obs, mb_raw_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_infos = [], [], [], [], [], [], []
    mb_neglogpacs, mb_next_obs = [], []
    extra_logs = defaultdict(list)
    init_dones = self.dones
    epinfos = []

    for env_step in range(self.nsteps):
      # 1. Policy step (See policies.py for step() function)
      obs = self.obs
      actions, values, neglogpacs = self.model.step(obs,
                                                    self.dones,
                                                    self.infos)
      '''
      Notations: d(t), s(t), info(t) -> a(t) ~ policy -> r(t) -> d(t+1), s(t+1), info(t+1)
      For mb_xxx[i], we save s(t), info(t), d(t) -> a(t) -> r(t)
      mb_obs[i]       : s(t)
      mb_infos[i]     : info(t) --> contains raw_obs(t), mask(t)
      mb_dones[i]     : done(t) --> if done(t) == 1: s(t) is initial state of an episode
      === self.model.step == (policy)
      mb_actions[i]   : a(t)
      mb_neglogpacs[i]: -log(pi(a(t)|s(t)))
      mb_values[i]    : V(s(t))
      === self.env.step == (environment)
      mb_reward[i]    : r(t)
      mb_next_dones[i]: d(t+1)
      mb_next_obs[i]  : s(t+1)
      mb_next_infos[i]: info(t+1) --> contains raw_obs(t+1), mask(t+1)
      '''
      assert self.obs.dtype == np.uint8
      assert self.infos[0]['observations'].dtype == np.uint8

      mb_obs.append(self.obs.copy())
      mb_actions.append(actions)
      mb_values.append(values)
      mb_neglogpacs.append(neglogpacs)
      mb_dones.append(self.dones)
      mb_infos.append(self.infos)
      mb_raw_obs.append(self.infos[0]['observations'])

      # 2. Environment step
      self.obs[:], rewards, self.dones, self.infos = self.env.step(actions)
      self.rewards = rewards

      # 3. Etc
      mb_next_obs.append(self.obs.copy())
      mb_rewards.append(rewards)

      # Record Episode statistics
      self._ep_return += rewards
      self._ep_length += 1

      for env_ind, (done, info) in enumerate(zip(self.dones, self.infos)):  # for each env: 0~11
        if done:
          epinfos.append({'l':self._ep_length[env_ind], 'r':self._ep_return[env_ind]})
          self._ep_length[env_ind] = 0
          self._ep_return[env_ind] = 0

    # 3. Post-processing
    ### batch of steps to batch of rollouts
    preprocess_mb_raw_obs = mb_raw_obs
    preprocess_mb_actions = mb_actions
    mb_raw_obs = np.asarray(mb_raw_obs, dtype=np.uint8)
    mb_obs = np.asarray(mb_obs, dtype=np.uint8)
    mb_next_obs = np.asarray(mb_next_obs, dtype=np.uint8)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    obs = self.obs
    last_values = self.model.value(obs, self.dones)

    # 3. Compute bonus
    bonus_dict = dict()
    mb_next_dones = np.concatenate((mb_dones[1:], [self.dones]), 0)

    if 'icm' in self.bonus_model.bonus_type:
      assert self.bonus_model.bonus_type == 'icm'
      assert self.model.state_encoder_net
      batch_size, num_env = mb_obs.shape[:2]
      flat_obs = _flatten(mb_obs)
      flat_next_obs = _flatten(mb_next_obs)
      flat_actions = _flatten(mb_actions)
      if self.bonus_model.icm_weight > 0:
        flat_bonus = self.model.getIntrinsicReward(flat_obs, flat_next_obs, flat_actions)
        flat_bonus = np.clip(flat_bonus, a_min=-1., a_max=1.)
        raw_bonus = _unflatten(flat_bonus, batch_size)
      else:
        raw_bonus = np.zeros( (batch_size, num_env))
      bonus_dict['icm'] = raw_bonus * self.bonus_model.icm_weight
      bonus_dict['bonus'] = bonus_dict['icm'].copy()
    elif ('sprl' in self.bonus_model.bonus_type):
      assert self.bonus_model.bonus_type == 'sprl'

      # * Bonus model
      # : We compute bonus based on "next" time step; we compare s(t+1-mdporder) and s(t+1).
      # : See get_bonus() in bonus_model.py for detail
      mb_next_infos = mb_infos[1:] + [self.infos]

      preprocess_mb_next_raw_obs = preprocess_mb_raw_obs[1:] + [self.infos[0]['observations']]
      mb_next_raw_obs = np.asarray(preprocess_mb_next_raw_obs, dtype=np.uint8)

      if self.bonus_model.scale_shortest_bonus == 0:
        sprl_bonus_dict = {
          'bonus': np.zeros_like(mb_rewards),
          'scale_shortest_bonus': 0,
          'shortest': np.zeros_like(mb_rewards),
          'raw_shortest': np.zeros_like(mb_rewards),
          }
      else:
        if self.rnet_trainer == None:
          sprl_bonus_dict = self.bonus_model.get_bonus(mb_next_raw_obs, mb_next_dones, mb_rewards, mb_next_infos, is_rnet_ready = True)
        else:
          sprl_bonus_dict = self.bonus_model.get_bonus(mb_next_raw_obs, mb_next_dones, mb_rewards, mb_next_infos, is_rnet_ready = self.rnet_trainer.algo_ready())

      bonus_dict['sprl'] = sprl_bonus_dict['bonus']
      bonus_dict['bonus'] = sprl_bonus_dict['bonus']
    elif ('eco' in self.bonus_model.bonus_type):
      assert self.bonus_model.bonus_type == 'eco'
      mb_next_infos = mb_infos[1:] + [self.infos]
      # * Bonus model
      # : We compute bonus based on "next" time step; we compare s(t+1-mdporder) and s(t+1).
      # : See get_bonus() in bonus_model.py for detail

      preprocess_mb_next_raw_obs = preprocess_mb_raw_obs[1:] + [self.infos[0]['observations']]
      mb_next_raw_obs = np.asarray(preprocess_mb_next_raw_obs, dtype=np.uint8)
      eco_bonus_dict = self.bonus_model.get_bonus(mb_next_raw_obs, mb_next_dones, mb_rewards, mb_next_infos, is_rnet_ready = self.rnet_trainer.algo_ready())

      bonus_dict['bonus'] = eco_bonus_dict['bonus']
      bonus_dict['eco'] = eco_bonus_dict['eco']
      bonus_dict['eco_clipped'] = eco_bonus_dict['eco']
    else:
      pass

    if 'bonus' in bonus_dict:
      mb_bonus = bonus_dict['bonus']
    else:
      mb_bonus = None

    if mb_bonus is not None:
      mb_rewards = mb_rewards + mb_bonus
      epinfos = self._update_accumulator(mb_next_dones, bonus_dict, epinfos, self.bonus_accum_dict)

    # 4. Update rnet_trainer
    if self.rnet_trainer is not None:
      self.rnet_trainer.update_buffer(preprocess_mb_raw_obs, preprocess_mb_actions, mb_rewards, mb_next_dones)

    # 5. Compute Return
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(self.nsteps)):
      if t == self.nsteps - 1:
        nextnonterminal = 1.0 - self.dones
        nextvalues = last_values
      else:
        nextnonterminal = 1.0 - mb_dones[t+1]
        nextvalues = mb_values[t+1]
      delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal -
               mb_values[t])
      mb_advs[t] = lastgaelam = (delta + self.gamma * self.lam *
                                 nextnonterminal * lastgaelam)
    mb_returns = mb_advs + mb_values
    return (map(sf01, (mb_obs, mb_next_obs, mb_returns,
                  mb_actions, mb_values, mb_neglogpacs)),
                  epinfos, extra_logs)

def sf01(arr):
  """Swap and then flatten axes 0 and 1.
  """
  s = arr.shape
  return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
  def f(_):
    return val
  return f

def learn(policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=3, gamma=0.99, lam=0.95,
          log_interval_per_update=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, load_path=None, eval_callback=None, cloud_sync_callback=None,
          cloud_sync_interval=1000, workdir='', summary_logger=None, use_curiosity=False,
          curiosity_strength=0.55, forward_inverse_ratio=0.96, curiosity_loss_strength=64,
          random_state_predictor=False, early_stop_timesteps=-1,
          early_rnet_train_freq=2, mid_rnet_train_freq=6, last_rnet_train_freq=12, early_rnet_steps=1000000):
  if isinstance(lr, float):
    lr = constfn(lr)
  else:
    assert callable(lr)
  if isinstance(cliprange, float):
    cliprange = constfn(cliprange)
  else:
    assert callable(cliprange)
  total_timesteps = int(total_timesteps)

  nenvs = env.num_envs
  ob_space = env.observation_space
  ac_space = env.action_space

  nbatch = nenvs * nsteps
  nbatch_train = nbatch // nminibatches
  nupdates = total_timesteps//nbatch
  if early_stop_timesteps > 0:
    early_stop_nupdates = min(early_stop_timesteps//nbatch, nupdates)
  else:
    early_stop_nupdates = nupdates
  tenth_update = max(early_stop_nupdates // 10, 1)
  assert nbatch % nminibatches == 0
  log_interval_per_update = max(log_interval_per_update, nupdates//500)

  # Get R-net trainer
  rnet_trainer = env.r_network_trainer

  ### R-net interval
  if rnet_trainer is not None:
    min_iter = min(rnet_trainer._buffer_size, 30000//nenvs ) // nsteps // 2 + 1
    default_interval = rnet_trainer._buffer_size // nsteps + 1
    early_iter = min_iter + int(early_rnet_steps//nbatch)
    if env.is_env['dmlab']:
      rnet_train_list = list(np.arange(min_iter, early_iter, early_rnet_train_freq)) # 2
      mid_iter = early_iter + int(early_rnet_steps//nbatch)
      rnet_train_list = rnet_train_list + list(np.arange(early_iter, mid_iter, mid_rnet_train_freq)) # 6
      rnet_train_list = rnet_train_list + list(np.arange(mid_iter, nupdates, last_rnet_train_freq)) # 12
    else:
      raise NotImplementedError

  pathak_multiplier = 20 if use_curiosity else 1

  make_model = lambda: Model(policy=policy, ob_space=ob_space,
                            ac_space=ac_space, nbatch_act=nenvs,
                            nbatch_train=nbatch_train, nsteps=nsteps,
                            vf_coef=vf_coef, ent_coef=ent_coef(0),
                            max_grad_norm=max_grad_norm,
                            use_curiosity=use_curiosity,
                            curiosity_strength=curiosity_strength,
                            forward_inverse_ratio=forward_inverse_ratio,
                            curiosity_loss_strength=curiosity_loss_strength,
                            random_state_predictor=random_state_predictor,
                            pathak_multiplier=pathak_multiplier)
  # pylint: enable=g-long-lambda
  if save_interval and workdir:
    with tf.io.gfile.GFile(osp.join(workdir, 'make_model.pkl'), 'wb') as fh:
      fh.write(dill.dumps(make_model))
  model = make_model()
  if load_path is not None:
    model.load(load_path)

  # Init bonus model
  bonus_model = BonusModel(
        num_envs=nenvs,
        emb_dim=env._embedding_size,
        obs_emb_fn=env._observation_embedding_fn,
        obs_compare_fn=env._observation_compare_fn,
        tolerance=env._tolerance,
        max_action_distance=env._max_action_distance,
        is_env=env.is_env)

  bonus_model.icm_weight = curiosity_strength
  # Construct Runner
  runner = Runner(env=env, model=model, bonus_model=bonus_model, rnet_trainer=rnet_trainer, rnet=env.r_net,
                  nsteps=nsteps, gamma=gamma, lam=lam, log_dir=summary_logger.log_dir,
                  eval_callback=eval_callback)

  epinfobuf = deque(maxlen=100)
  last_epinfobuf = deque(maxlen=nenvs)
  tstart = time.time()

  for update in range(1, nupdates+1):
    frac = 1.0 - (update - 1.0) / nupdates
    entnow = ent_coef(1.0 - frac)
    lrnow = lr(frac)
    cliprangenow = cliprange(frac)

    # 1. Rollout policy 128 steps & compute bonus
    (obs, next_obs, returns, actions, values,
     neglogpacs), epinfos, extra_logs = runner.run()
    epinfobuf.extend(epinfos)
    last_epinfobuf.extend(epinfos)

    mblossvals = []
    # 2. PPO update
    inds = np.arange(nbatch)
    for _ in range(noptepochs):
      np.random.shuffle(inds)
      for start in range(0, nbatch, nbatch_train):
        end = start + nbatch_train
        mbinds = inds[start:end]
        train_args = {'lr': lrnow, 'cliprange': cliprangenow, 'obs': obs[mbinds], 'next_obs': next_obs[mbinds],
                      'returns': returns[mbinds], 'actions': actions[mbinds],'values': values[mbinds],
                      'neglogpacs': neglogpacs[mbinds],}
        mblossvals.append(model.train(**train_args))

    # 3. Train R-net
    if rnet_trainer is not None and (update in rnet_train_list) and rnet_trainer.train_ready():
      rnet_trainer.train(update,nupdates)

    # 4. Eval
    lossvals = np.mean(mblossvals, axis=0)
    if update % log_interval_per_update == 0 or update == 1:
      assert update*nbatch == runner._collection_iteration * runner.nenvs * runner.nsteps
      global_step = update * nbatch
      summary_logger.set_step( global_step )

      ### 4-2. Logging the training stats
      duration = time.time() - tstart
      fps = update*nsteps*nenvs / duration
      #
      if rnet_trainer is not None:
        summary_logger.logkv('r_network/Accuracy (train)', rnet_trainer.get_train_accuracy())
        summary_logger.logkv('r_network/Accuracy (valid)', rnet_trainer.get_valid_accuracy())
      #
      if bonus_model._use_icm:
        summary_logger.logkv('episode/(Bonus) ICM', safemean([epinfo['icm'] for epinfo in last_epinfobuf]))
      if bonus_model._use_sprl:
        summary_logger.logkv('episode/(Bonus) SPRL', safemean([epinfo['sprl'] for epinfo in last_epinfobuf]))
      if bonus_model._use_eco:
        summary_logger.logkv('episode/(Bonus) ECO', safemean([epinfo['eco'] for epinfo in last_epinfobuf]))
      #
      summary_logger.logkv('episode/Avg.Return (train)', safemean([epinfo['r'] for epinfo in epinfobuf]))
      summary_logger.logkv('episode/length (train)', safemean([epinfo['l'] for epinfo in last_epinfobuf]))
      summary_logger.logkv('steps/total_timesteps', update*nbatch)
      summary_logger.logkv('time/Fps', fps)

      for k, v in extra_logs.items():
        summary_logger.logkv('%s'%(k), sum(v)/len(v))

      summary_logger.dumpkvs()

      if early_stop_timesteps > 0 and early_stop_timesteps < global_step:
        break

    if (cloud_sync_interval and update % cloud_sync_interval == 0 and
        cloud_sync_callback):
      cloud_sync_callback()

  env.close()
  summary_logger.close()
  return model

def safemean(xs):
  return np.mean(xs) if xs else np.nan
