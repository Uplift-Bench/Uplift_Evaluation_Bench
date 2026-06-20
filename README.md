# Evaluating Uplift Modeling under Structural Biases: Insights into Metric Stability and Model Robustness

*This paper is accepted by KDD 2026.*

### Requirements

- Python 3.9+
- CUDA 12.6 (GPU acceleration, optional)

### Installation

```bash
# The repository is relatively large, so cloning may take a while
git clone https://github.com/Uplift-Bench/Uplift_Evaluation_Bench.git

cd Uplift_Evaluation_Bench

# Create and activate conda virtual environment
conda create -n uplift python=3.11 -y
conda activate uplift

# Install dependencies (default: PyTorch 2.6.0 with CUDA 12.6)
pip install -r requirements.txt
```


## Models

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
# Run DGP for Criteo dataset
python Criteo/DGP.py
```

Generated data is saved in the corresponding bias subdirectories under `data/Bank/`, `data/Hillstrom/`, or `data/Criteo/`.

### Run Experiments

Run the main experiment scripts to train and evaluate all models:

```bash
cd Uplift_Evaluation_Bench
# Run main experiment for Bank dataset
python Bank/main_experiment_Bank.py
# Run main experiment for Hillstrom dataset
python Hillstrom/main_experiment_Hillstrom.py
# Run main experiment for Criteo dataset
python Criteo/main_experiment_Criteo.py
```

`output_data/` already contains our original experiment output data, which can be used directly for result comparison and analysis.

> **Note**: Re-running experiments will automatically overwrite the result files under `output_data/`. If you need to compare with the original data, please back up this directory first.

## Reproducibility

All datasets have fixed the main random seeds, but Optuna hyperparameter search itself has inherent randomness (e.g., TPE sampling). The optimal hyperparameters selected between different runs may vary slightly, resulting in minor fluctuations in the final results. Some models' numerical values and rankings may show slight fluctuations, but the overall trend conclusions under each bias scenario remain unaffected.

## Appendix

The `Appendix/` folder contains supplementary materials for the paper.

## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| Bank | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) | Bank marketing dataset |
| Hillstrom | [Kevin Hillstrom E-Mail Analytics](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) | E-commerce email marketing dataset |
| Criteo | [Criteo Large-Scale ITE Benchmark](https://github.com/criteo-research/large-scale-ITE-UM-benchmark) | Large-scale uplift modeling dataset |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yang2026evaluating,
  title={Evaluating Uplift Modeling under Structural Biases: Insights into Metric Stability and Model Robustness},
  author={Yang, Yuxuan and Liu, Dugang and Huang, Yiyan},
  journal={arXiv preprint arXiv:2603.20775},
  year={2026}
}
```



