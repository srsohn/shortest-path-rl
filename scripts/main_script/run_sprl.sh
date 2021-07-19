default="--workdir=./dmlab --use_level_cache --num_env=12"
ALGO="--policy_archi=cnn --method=sprl --use_various_tol"
INTERVALS="--num_timesteps=20000000"
PARAMS="$default $ALGO $INTERVALS"
argument="$@"
echo "argument = $argument"

#### Hparams ####
seed=123
scenario='sparse'
lr=0.0003
ent=0.004
bias=0.5
tol=1
k=10
scale=0.06
neg_th=20
t_num=200
bonus_thres=90
neg_upper_ratio=0.1
rnet_lr=3.0e-4
rnet_num_epochs=1
rnet_train_size=30000
rnet_buffer_size=60000
early_rnet_steps=1000000
rnet_train_interval=2
l2_norm=0.0001
post_fix='--max_grad_norm 10'

#### Run ####
python $(dirname "$0")/../launcher_script.py \
$PARAMS --scenario=$scenario \
--policy_lr $lr --policy_ent_coef $ent --seed $seed \
--tolerance $tol --max_action_distance=$k \
--scale_shortest_bonus $scale --bonus_bias=$bias --neg_sample_adder=$neg_th \
--bonus_thres=$bonus_thres --tol_num=$t_num \
--neg_upper_ratio=$neg_upper_ratio --rnet_train_interval=$rnet_train_interval \
--rnet_lr=$rnet_lr --rnet_num_epochs=$rnet_num_epochs \
--rnet_train_size=$rnet_train_size --rnet_buffer_size=$rnet_buffer_size \
--early_rnet_steps=$early_rnet_steps --l2_norm=$l2_norm $post_fix $argument
