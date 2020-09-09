# Multi-Actor-Attention-Critic
Code for [*Actor-Attention-Critic for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1810.02912) (Iqbal and Sha, ICML 2019)

## Requirements
* Python 3.6.1 (Minimum)
* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

The versions are just what I used and not necessarily strict requirements.

## How to Run

All training code is contained within `main.py`. To view options simply run:

```shell
python main.py --help
```
The "Cooperative Treasure Collection" environment from our paper is referred to as `fullobs_collect_treasure` in this repo, and "Rover-Tower" is referred to as `multi_speaker_listener`.

In order to match our experiments, the maximum episode length should be set to 100 for Cooperative Treasure Collection and 25 for Rover-Tower.

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@InProceedings{pmlr-v97-iqbal19a,
  title =    {Actor-Attention-Critic for Multi-Agent Reinforcement Learning},
  author =   {Iqbal, Shariq and Sha, Fei},
  booktitle =    {Proceedings of the 36th International Conference on Machine Learning},
  pages =    {2961--2970},
  year =     {2019},
  editor =   {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume =   {97},
  series =   {Proceedings of Machine Learning Research},
  address =      {Long Beach, California, USA},
  month =    {09--15 Jun},
  publisher =    {PMLR},
  pdf =      {http://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf},
  url =      {http://proceedings.mlr.press/v97/iqbal19a.html},
}
```
