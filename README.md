# playertracking
```shell
cd
nano .condarc
```
```yaml
channels:
  - pytorch
  - conda-forge
  - anaconda
  - defaults
channel_priority: strict
```

```shell
conda create -yn <env_name> python=3.8
conda activate <env_name>
```
MacOS:
```shell
conda install pytorch torchvision torchaudio -yc pytorch
```
Linux Cuda 11.6:
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```
```shell
conda install -y numpy opencv tensorflow matplotlib pandas tqdm scikit-learn pyyaml seaborn onnx
```

Then Run:
```shell
python track.py
```