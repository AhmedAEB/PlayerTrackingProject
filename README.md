# Player Tracking Project

## Installation

### Step 1: Install Anaconda and Setup condarc as follows:
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

### Step 2: Create conda environment
```shell
conda create -yn <env_name> python=3.10
conda activate <env_name>
```

### Step 3: Install pytorch (based on your system using pip)
https://pytorch.org/get-started/locally/

### Step 4: Install Yolo
```pip3 install ultralytics```

### Step 5: Install tensorflow
```pip3 install tensorflow scikit-learn```

### Step 5: Run & Enjoy ☺️
```shell
python3 main-gui.py
```