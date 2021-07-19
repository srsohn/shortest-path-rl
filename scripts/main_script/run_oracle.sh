default="--workdir=./dmlab --use_level_cache --num_env=12"
ALGO="--policy_archi=cnn --method=oracle --use_various_tol"
INTERVALS="--num_timesteps=20000000"
PARAMS="$default $ALGO $INTERVALS"
argument="$@"
echo "argument = $argument"

#### Hparams ####
seed=123
scenario='sparse'
lr=0.00025
ent=0.0066
post_fix=''

#### Run ####
python $(dirname "$0")/../launcher_script.py \
$PARAMS --scenario=$scenario \
--policy_lr $lr --policy_ent_coef $ent --seed $seed \
$post_fix $argument