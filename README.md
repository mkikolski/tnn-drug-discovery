# Reinforcement Learning Docking Informed Drug Discovery Framework

## Installation

This framework was tested on Ubuntu 22.04 with Python 3.10 and requires around 16 GB of RAM and CUDA compatible GPU to run.

### Install the prerequisities

```bash
sudo apt update
sudo apt install openbabel
sudo apt-get update 
sudo apt-get install ocl-icd-opencl-dev clinfo libnetcdf-dev
```

### Set the stack size required for QVina GPU

```bash
ulimit -s 8192
```

### Install fpocket

```bash
git clone https://github.com/Discngine/fpocket.git
cd fpocket
make
sudo make install
cd ..
```

### Install QuickVina2-GPU

Follow the instructions at (https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1/tree/main/QuickVina2-GPU-2.1)[https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1/tree/main/QuickVina2-GPU-2.1]

### Install the framework

```bash
git clone https://github.com/mkikolski/tnn-drug-discovery.git
cd tnn-drug-discovery
pip install -r requirements.txt
```

## Running the code

First use `fpocket` to locate the binding site of your PDB file

```bash
fpocket -f your.pdb
```

Create a copy of `example_config.toml` and set the required values according to your system specifications and detected pocket details.

The framework supports folowing steps:
- `pretrain` for pretraining the TNN generator
- `reward` for computing QED, SA and docking scores for pretraining the DQN
- `rl` for fine tuning the generator with the DQN
- `generate` for generating the molecules using pretrained generator

The operations are executed like that:
```bash
python run.py --step <one of the steps above> --config <path to your config>
```