# [Shortest-Path Constrained Reinforcement Learning for Sparse Reward Tasks](https://arxiv.org/abs/2107.06405)

In ICML 2021.

[Sungryull Sohn¹²*](https://sites.google.com/view/sungryull),
[Sungtae Lee³*](https://sites.google.com/yonsei.ac.kr/sungtae/),
[Jongwook Choi¹](https://wook.kr/),
[Harm van Seijen⁴](https://www.microsoft.com/en-us/research/people/havansei/),
[Mehdi Fatemi⁴](https://www.microsoft.com/en-us/research/people/mefatemi/),
[Honglak Lee²¹](https://web.eecs.umich.edu/~honglak/) <br/>
*: Equal Contributions <br/>
¹University of Michigan, ²LG AI Research, ³Yonsei University, ⁴Microsoft Research <br/>

This is an official implementation of our ICML 2021 paper [Shortest-Path Constrained Reinforcement Learning for Sparse Reward Tasks](https://arxiv.org/abs/2107.06405).
For simplicity of the code, we only included codes for DeepMind Lab environment.
All algorithms used in the experiments (SPRL, ECO, ICM, PPO, Oracle) can be run using this code.

### Requirements

The code was tested on Linux only. We used python 3.6.

### Installation

These installation steps are based on [episodic_curiosity](https://github.com/google-research/episodic-curiosity) repo from Google Research.

Clone this repository:

```shell
git clone https://github.com/srsohn/shortest-path-rl.git
cd shortest-path-rl
```

We need to use a modified version of [DeepMind Lab](https://github.com/deepmind/lab):

Clone DeepMind Lab:

```shell
git clone https://github.com/deepmind/lab 
cd lab
```

Apply our patch to DeepMind Lab:

```shell
git checkout 7b851dcbf6171fa184bf8a25bf2c87fe6d3f5380
git checkout -b modified_dmlab
git apply ../third_party/dmlab/dmlab_min_goal_distance.patch
```

Install dependencies of DMLab from [deepmind_lab](https://github.com/deepmind/lab/blob/master/docs/users/build.md#lua-and-python-dependencies) repository.

Right before building deepmind_lab,
<br />&nbsp;&nbsp;&nbsp; - Replace `BUILD`, `WORKSPACE` files in `lab` directory with the files in `changes_in_lab` directory.
<br />&nbsp;&nbsp;&nbsp; - Copy `python.BUILD` from `changes_in_lab` directory to `lab` directory.

Please make sure
<br />&nbsp;&nbsp;&nbsp; - `lab/python.BUILD` has correct path to python. Please refer to `changes_in_lab/python.BUILD`.
<br />&nbsp;&nbsp;&nbsp; - `lab/WORKSPACE` has correct path to python system in its last code line (new_local_repository).

Then build DeepMind Lab (in the `lab/` directory):
```shell
bazel build -c opt //:deepmind_lab.so
```

Once you've installed DMLab dependencies, you'll need to run:

```shell
bazel build -c opt python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py3-none-any.whl --force-reinstall
```

Finally, install episodic curiosity and its pip dependencies:

```shell
cd shortest-path-rl
pip install --user -e .
```

Please refer to `changes_in_lab/lab_install.txt` for the summary of deepmind_lab & sprl installation.

### Training

To run the training code, go to base directory and run the script below.

```shell
sh scripts/main_script/run_{eco,icm,oracle,ppo,sprl}.sh
```

`seed_set` controls the random seed and `scenario_set` denotes the environment to run. 
Hyperparameters of the algorithms can be changed in the bash file in the `scripts/main_script` directory.

`scripts/launcher_script.py` is the main entry point.

Main flags:

| Flag | Descriptions |
| :----------- | :--------- |
| --method | Solving method to use. Possible values: `ppo, icm, eco, sprl, oracle` |
| --scenario | Scenario to launch. Possible values: `GoalSmall, ObjectMany, GoalLarge` |
| --workdir | Directory where logs and checkpoints will be stored.  |

Training usually takes a couple of days.
Using GTX1080 Titan X, fps and VRAM usage are as below.

| Algorithm | Fps | VRAM |
| :----------- | :--------- | :--------- |
| ppo | ~ 900 | 1.4G |
| icm | ~ 500 | 4.5G |
| eco | ~ 200 | 4.5G |
| sprl | ~ 200 | 4.5G |
| oracle | ~ 800 | 1.4G |

Under the hood,
[launcher_script.py](https://github.com/srsohn/shortest-path-rl/blob/master/scripts/launcher_script.py)
launches
[train_policy.py](https://github.com/srsohn/shortest-path-rl/blob/master/sprl/train_policy.py)
with the right hyperparameters. [train_policy.py](https://github.com/srsohn/shortest-path-rl/blob/master/sprl/train_policy.py) generates environment and run [ppo2.py](https://github.com/srsohn/shortest-path-rl/blob/master/third_party/baselines/ppo2/ppo2.py) where rolling out the transitions take place.

### Sample Result

In `workdir/`, tensorflow logs will be saved. Run tensorboard to visualize training logs.
Training logs vary depending on the algorithm including below logs.
In `sample_result/`, sample training logs from tensorboard are available.

<img src="../master/sample_result/SampleResult_AvgReturn.png?raw=true" width="230"> <img src="../master/sample_result/SampleResult_Bonus.png?raw=true" width="230"> <img src="../master/sample_result/SampleResult_Fps.png?raw=true" width="230"> <img src="../master/sample_result/SampleResult_PolicyLoss.png?raw=true" width="230"> <img src="../master/sample_result/SampleResult_RNetTrainingAcc.png?raw=true" width="230">

### Citation

If you use this work, please cite:

```
@inproceedings{sohn2021shortest,
  title={Shortest-Path Constrained Reinforcement Learning for Sparse Reward Tasks},
  author={Sohn, Sungryull and Lee, Sungtae and Choi, Jongwook and Van Seijen, Harm H and Fatemi, Mehdi and Lee, Honglak},
  booktitle={International Conference on Machine Learning},
  pages={9780--9790},
  year={2021},
  organization={PMLR}
}
```

### License

Apache License
