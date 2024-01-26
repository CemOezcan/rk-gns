# Recurrent Kalman Graph Network Simulators
This repository contains the implementation of Recurrent Kalman Graph Network Simulators (RK-GNS), 
a novel Graph Neural Networks-based physics simulator capable of inferring a task-specific forward dynamics model 
by estimating crucial system parameters from sensor observations. 

## Evaluation Results
Here, we provide the results of our [Self-Supervised](https://api.wandb.ai/links/cem_oezcan/pdk22ejv), 
[Supervised](https://wandb.ai/cem_oezcan/RK-GNS/reports/Supervised-Results--Vmlldzo2NjQ3MDM5), [Baselines](https://api.wandb.ai/links/cem_oezcan/kkfcadio) and 
[Poisson](https://api.wandb.ai/links/cem_oezcan/henynbhy) experiments. For simplicity, these reports only contain the metrics analyzed in the thesis, 
i.e. those with respect to the 50-step error. To run your own experiments, follow the instructions below.
## Repository Structure
```
.             
├── config                     # Config files to set up experiments.
├── data                       # Datasets.
    ├── deformable_plate                     
        ├── input               # Must contain the deformable_plate dataset.
        ├── output              # Contains model snapshots, evaluation results and visualizations.
├── src                        # Source code.
    ├── algorithms              # Model training and testing.
    ├── data                    # Data preprocessing and management.
    ├── model                   # System models.
    ├── modules                 # Neural network modules.
    ├── util                    # Utility functions and classes.
├── LICENSE                    
├── README.md
├── main.py                    # The main script. Execute this to start experiments.             
└── requirements.txt
```           

## Downloading Datasets
We train our models on the `deformable_plate` dataset, which can be downloaded [here](https://drive.google.com/drive/folders/1hNQyOSWE8PncOoLwXku7-gTLQQlwcOrW?usp=sharing).
Move the datasets to the `rk-gns/data/deformable_plate` subdirectory and rename them to `deformable_plate_eval.pkl`, `deformable_plate_test.pkl` and `deformable_plate_train.pkl` respectively.

## Setup

### Virtual Environment
Use [virtualenv](https://virtualenv.pypa.io/) to create a virtual environment called `venv` in this projects source directory by running 
```bash
python -m venv -p <path/to/right/python/executable> <path/to/project/venv>
```
We are using `Python 3.9.7`.
To install all dependencies, execute
```bash
source venv/bin/activate
```
to activate the virtual environment `./venv` you just created and run
```bash
pip install -r requirements.txt
```
to install all necessary packages using [PyPI](https://pypi.org/).

### Logging
We provide logging of metrics and visualizations to [W&B](https://wandb.ai). 
This requires logging in to [W&B](https://wandb.ai) via 
```bash
wandb login
```
For more information, read the [quickstart guide of W&B](https://docs.wandb.ai/quickstart).

## Running Experiments
For simplicity, we set up different Branches for different types of experiments.
```
Poisson              # Models that predict the Roisson's ratio. 
Baselines            # Baseline models such as MGN, MGN (M), GGNS, GGNS (N), GGNS (R)
Supervised           # Supervised models such as GR-GNS, GR-GNS (N), GR-GNS (R), GR-GGNS, GR-GGNS (N), GR-GGNS (R)
Self-Supervised      # Self-supervised explicit models GR-GNS, GR-GNS (N), GR-GNS (R), GR-GGNS, GR-GGNS (N), GR-GGNS (R)
RKN_Supervised       # Supervised RKN models such as RK-GNS, RK-GGNS
RKN_Self-Supervised  # Self-supervised latent variable RKN models such as RK-GNS, RK-GNS (N), RK-GNS (R), RK-GGNS, RK-GGNS (N), RK-GGNS (R)
```
Config files, like for example `mgn.yaml`, in `./config` define the experimental setup.

### Standard
First, run 
```bash
source venv/bin/activate
```
to activate the virtual environment you previously created, and then
```bash
python main.py "mgn"
```
to run the experiment defined in `./config/mgn.yaml`. Note that you can substitute `mgn.yaml` by any config file in `./config`.

### BwUniCluster
To run experiments on the [BwUniCluster](https://wiki.bwhpc.de/e/Category:BwUniCluster_2.0), execute
```bash
sbatch -p gpu_8 -n 1 -t 10:00:00 --gres=gpu:1 job_mgn.sh
```
where `job_mgn.sh` is the corresponding bash script to the config file `mgn.yaml`. 
Note that you can adjust the allocation time by modifying `-t HH:MM:SS`. The following list contains rough run-time 
estimates for different types of models trained on [NVIDIA Tesla V100-GPUs](https://www.nvidia.com/de-de/data-center/tesla-v100/).
```
RK-GGNS/GR-GGNS: 18:00:00
RK-GNS/GR-GNS: 18:00:00

SS_RK-GGNS/SS_GR-GGNS: 16:00:00
SS_RK-GNS/SS_GR-GNS: 15:00:00

Baselines: 06:00:00
Poisson: 12:00:00
```
Training progress will be saved periodically after n epochs, n being an adjustable experimental parameter. 
Hence, allocating less time when queueing issues arise is an option.
