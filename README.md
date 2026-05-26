## Quick Start

### Requirements

- Python 3.9+
- CUDA 12.6 (GPU acceleration, optional)

### Installation

```bash
# The repository is relatively large, so cloning may take a while
git clone https://github.com/YxuanYang/Uplift_Evaluation_Bench.git

cd Uplift_Evaluation_Bench

# Create and activate conda virtual environment
conda create -n uplift python=3.11 -y
conda activate uplift

# Install dependencies (default: PyTorch 2.6.0 with CUDA 12.6)
pip install -r requirements.txt
```

## Project Structure

```
Uplift_Evaluation_Bench/
├── Bank/                            
│   ├── DGP.py                       
│   ├── main_experiment_Bank.py      
│   ├── meta_learner.py              
│   ├── Net_optimization_Bank.py     
│   ├── Net_model/                   
│   │   ├── main_TAR.py              
│   │   └── base_net.py              
│   └── Dragonnet/                
│       ├── Dragonnet_model.py       
│       └── main_Dragon.py           
├── Hillstrom/                       
│   ├── DGP.py
│   ├── main_experiment_Hillstrom.py
│   ├── meta_learner.py
│   ├── Net_optimization_Hillstrom.py
│   ├── Net_model/
│   └── Dragonnet/
├── data/                            
│   ├── Bank/
│   │   ├── bank-additional-full.csv # Raw data
│   │   ├── hidden/                  
│   │   ├── measurement/            
│   │   ├── selection/              
│   │   └── spillover/             
│   └── Hillstrom/
│       ├── Kevin_Hillstrom_...csv   # Raw data
│       ├── hidden/
│       ├── measurement/
│       ├── selection/
│       └── spillover/
├── output_data/                     # Our original experiment output data
│   ├── Bank/
│   │   ├── hidden/
│   │   ├── measurement/
│   │   ├── selection/
│   │   └── spillover/
│   └── Hillstrom/
│       ├── hidden/
│       ├── measurement/
│       ├── selection/
│       └── spillover/
├── requirements.txt
└── README.md
```

## Supported Models

### Meta-Learners

S-Learner, T-Learner, X-Learner, R-Learner, U-Learner, DR-Learner, RA-Learner

### Neural Network Models

TARNet, DragonNet


## Bias Types

This framework introduces the following four types of bias through the DGP module, with adjustable bias intensity:

| Bias Type | Parameter | Description |
|-----------|-----------|-------------|
| Selection Bias | `xi` | Treatment assignment depends on covariates |
| Hidden Confounding | `m` | Unobserved confounding variables exist |
| Measurement Error | `select_omega` | Outcome variable contains measurement noise |
| Spillover Effect | `beat_ny_0`, `beat_ny_1` | Treatment effects spill over between individuals |



## Usage

### Data Generation

Run DGP scripts to generate experimental data with different biases:

```bash
cd Uplift_Evaluation_Bench
# Run DGP for Bank dataset
python Bank/DGP.py
# Run DGP for Hillstrom dataset
python Hillstrom/DGP.py
```

Generated data is saved in the corresponding bias subdirectories under `data/Bank/` or `data/Hillstrom/`.

### Run Experiments

Run the main experiment scripts to train and evaluate all models:

```bash
cd Uplift_Evaluation_Bench
# Run main experiment for Bank dataset
python Bank/main_experiment_Bank.py
# Run main experiment for Hillstrom dataset
python Hillstrom/main_experiment_Hillstrom.py
```

`output_data/` already contains our original experiment output data, which can be used directly for result comparison and analysis.

> **Note**: Re-running experiments will automatically overwrite the result files under `output_data/`. If you need to compare with the original data, please back up this directory first.

## Reproducibility

Both datasets have fixed the main random seeds, but Optuna hyperparameter search itself has inherent randomness (e.g., TPE sampling). The optimal hyperparameters selected between different runs may vary slightly, resulting in minor fluctuations in the final results.

- **Bank dataset**: Some models' numerical values and rankings may show slight fluctuations, but the overall trend conclusions under each bias scenario remain unaffected.
- **Hillstrom dataset**: Some models' numerical values and rankings may show slight fluctuations, but the overall trend conclusions under each bias scenario remain unaffected.

## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| Bank | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) | Bank marketing dataset |
| Hillstrom | [Kevin Hillstrom E-Mail Analytics](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) | E-commerce email marketing dataset |

## License


