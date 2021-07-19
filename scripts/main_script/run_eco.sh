default="--workdir=./dmlab --use_level_cache --num_env=12"
ALGO="--policy_archi=cnn --method=eco --use_various_tol"
INTERVALS="--num_timesteps=20000000 --rnet_train_interval=2 --rnet_buffer_size=60000"
PARAMS="$default $ALGO $INTERVALS"
argument="$@"
echo "argument = $argument"

#### Hparams ####
seed=123
scenario='sparse'
lr=0.00025
ent=0.0021
bias=0.5
tol=1
k=5
scale=0.03
neg_th=20
epmem_size=200
bonus_thres=90
neg_upper_ratio=0.1
rnet_lr=3.0e-4
rnet_num_epochs=1
rnet_train_size=30000
rnet_buffer_size=60000
early_rnet_steps=1000000
l2_norm=0.0001
post_fix=''

#### Run ####
python $(dirname "$0")/../launcher_script.py \
$PARAMS --scenario=$scenario \
--policy_lr $lr --policy_ent_coef $ent --seed $seed \
--tolerance $tol --max_action_distance=$k \
--scale_eco_bonus $scale --bonus_bias=$bias --neg_sample_adder=$neg_th \
--bonus_thres=$bonus_thres --epmem_size=$epmem_size \
--neg_upper_ratio=$neg_upper_ratio \
--rnet_lr=$rnet_lr --rnet_num_epochs=$rnet_num_epochs \
--rnet_train_size=$rnet_train_size --rnet_buffer_size=$rnet_buffer_size \
--early_rnet_steps=$early_rnet_steps --l2_norm=$l2_norm $post_fix $argument