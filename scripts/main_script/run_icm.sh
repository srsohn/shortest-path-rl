default="--workdir=./dmlab --use_level_cache --num_env=12"
ALGO="--policy_archi=cnn --method=icm --use_various_tol"
INTERVALS="--num_timesteps=20000000"
PARAMS="$default $ALGO $INTERVALS"
argument="$@"
echo "argument = $argument"

#### Hparams ####
seed=123
scenario='sparse'
lr=0.00025
ent=0.0042
forward_inverse_ratio=0.96
curiosity_loss_strength=64
curiosity_strength=0.55
post_fix=''

#### Run ####
python $(dirname "$0")/../launcher_script.py \
$PARAMS --scenario=$scenario \
--policy_lr $lr --policy_ent_coef $ent --seed $seed \
--forward_inverse_ratio $forward_inverse_ratio \
--curiosity_loss_strength $curiosity_loss_strength --curiosity_strength $curiosity_strength \
$post_fix $argument