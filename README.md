# Probabilistic Cascading Diffusion Models
![](https://github.com/lihenryhfl/pcdm/blob/main/pcdm.gif)

Official codebase for Likelihood Training of Cascaded Diffusion Models via Hierarchical Volume-preserving Maps (https://openreview.net/forum?id=sojpn00o8z). Code is based on the codebase for Variational Diffusion Models.

## Setup: Installing required libraries

```
pip install --upgrade -r requirements.txt
```

## Train/evaluate: CIFAR-10

The commands below assume that the code is checked out into the `./pcdm` directory.

To evaluate from a pre-trained checkpoint:
```
python3 -m vdm.main --mode=eval --config=pcdm/configs/cifar10.py --workdir=[workdir] --checkpoint=[checkpoint]'
```
where `[workdir]` is a directory to write results to, such as `'/tmp/vdm-workdir'`. Running the command above will print out a bunch of statistics, including `eval_bpd=2.637`, which matches the result in the paper (2.35).

To train:
```
python3 -m pcdm.main --config=pcdm/configs/cifar10.py --workdir=[workdir]
```

## Train/evaluate: ImageNet 32x32, 64x64, 128x128

Run the commands:
```
python3 -m pcdm.main --config=pcdm/configs/imagenet32.py --workdir=[workdir]
```

```
python3 -m pcdm.main --config=pcdm/configs/imagenet64.py --workdir=[workdir]
```

```
python3 -m pcdm.main --config=pcdm/configs/imagenet128.py --workdir=[workdir]
```
respectively.
