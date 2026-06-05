# Shallow Neural Networks for Signal Decomposition

This repository contains the code, dashboard, training scripts, and evaluation utilities for the paper:

**Shallow Neural Networks for Signal Decomposition**  
Vladimir Bortsov

The local paper PDF is included as:

```text
Shallow_Neural_Networks_for_Signal_Decomposition.pdf
```

The project studies whether shallow Conv1D-style neural networks can decompose synthetic one-dimensional signals into source components while using far fewer parameters than deeper baselines such as TasNet, U-Net, SepFormer, and recurrent models.

The main benchmark uses synthetic signals with known ground-truth components. The default three-component setup contains harmonic, AM-FM, and chirp sources. The extended setup adds trend and transient components.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app/` | Streamlit dashboard for model comparison, inference, research plots, signal generation, real-data tests, and exports. |
| `decomposers/ML_methods/NN_based/` | Neural model implementations, training scripts, shared training utilities, dataset wrappers, and experiment catalog. |
| `decomposers/ML_methods/NN_based/experiment_catalog.py` | Central model registry used by training, evaluation, and dashboard metadata. |
| `evaluation/` | Metric code, decomposition scoring, model evaluation runners, CHB-MIT/real-data tests, and generated comparison artifacts. |
| `generators/` | Synthetic signal generation utilities used by training and testing. |
| `RealData/` | Local real-signal data folder. Large datasets should stay local unless intentionally versioned. |
| `docs/FILE_GUIDE.md` | Concise file-by-file guide for maintained source files. |

## Quick Start

Use Python 3.9 or newer. The repo currently does not provide a locked dependency file, so install the packages used by the training and dashboard code directly:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install torch numpy pandas matplotlib streamlit scipy scikit-learn
```

Start the dashboard:

```powershell
py -m streamlit run app/main.py
```

The dashboard includes pages for reconstruction inspection, model comparison, research comparison, data-efficiency analysis, training history, hard-case/error analysis, synthetic signal generation, and export assets.

## Main Results From The Paper

The paper compares 14 architectures. The RNN baseline is trained and reported separately because it is unstable in the main benchmark conditions.

Important reported outcomes:

| Result | Paper value |
| --- | --- |
| Best overall model | TasNet |
| TasNet macro correlation | 0.887 |
| TasNet macro SNR | 8.97 dB |
| Best shallow model | Attention-Stem Multi-Head Multiscale TCN |
| Best shallow macro correlation | 0.811 |
| Best shallow parameter count | 78,070 |
| Parameter reduction vs TasNet | about 32x |
| Inference latency reduction vs TasNet | about 2.6x |

The paper's main conclusion is not that shallow models always beat deep separators. It shows that a carefully designed shallow model can get close to deep-model performance with a much smaller parameter budget, especially on structured synthetic decomposition tasks.

## Rerunning Experiments

All model training goes through the unified launcher:

```powershell
py decomposers/ML_methods/NN_based/run_train.py <model_key> [training args]
```

All model evaluation goes through:

```powershell
py evaluation/model_eval/run_eval.py <model_key|all> [evaluation args]
```

The model keys are defined in:

```text
decomposers/ML_methods/NN_based/experiment_catalog.py
```

The dashboard model metadata is defined in:

```text
app/config/model_registry.json
```

### Train The Best Shallow Model

Paper-scale full-pool training uses 8000 training samples, 400 validation samples, 400 test samples, 20 epochs, and seed 42:

```powershell
py decomposers/ML_methods/NN_based/run_train.py attention_stem_multi_head_multiscale_tcn --train-samples 8000 --val-samples 400 --test-samples 400 --epochs 20 --seed 42
```

Many training scripts currently default to `--train-samples 4000` for faster local runs. To reproduce the paper's full-pool setting, pass `--train-samples 8000` explicitly.

### Train TasNet

TasNet uses a smaller batch size in the paper because it is heavier than the shallow models:

```powershell
py decomposers/ML_methods/NN_based/run_train.py tasnet --train-samples 8000 --val-samples 400 --test-samples 400 --epochs 20 --batch-size 16 --seed 42
```

### Train SepFormer

SepFormer is the heaviest baseline and uses batch size 8 in the paper setup:

```powershell
py decomposers/ML_methods/NN_based/run_train.py sepformer --train-samples 8000 --val-samples 400 --test-samples 400 --epochs 20 --batch-size 8 --seed 42
```

### Evaluate One Model

```powershell
py evaluation/model_eval/run_eval.py attention_stem_multi_head_multiscale_tcn --test-samples 400
```

### Evaluate All Cataloged Models

```powershell
py evaluation/model_eval/run_eval.py all --test-samples 400
```

`run_eval.py all` evaluates models that have evaluation specs in `experiment_catalog.py`. Some experimental or disabled variants may not have standalone evaluation scripts. Those can still be inspected through saved reports or by adding an evaluation spec.

### Generate Comparison Tables And Plots

After training/evaluating models, regenerate comparison artifacts:

```powershell
py evaluation/model_eval/6compare_all_evals.py --skip-run
```

The generated files are written under:

```text
evaluation/model_eval/artifacts/
```

Typical outputs include:

```text
model_comparison_table.md
macro_corr_comparison.png
macro_snr_comparison.png
test_loss_comparison.png
eval_status.json
```

## Experiment Map

| Paper experiment | What it does | Where to rerun or inspect it |
| --- | --- | --- |
| E1 Main benchmark | Trains and compares deep and shallow models on the three-component synthetic task. | `run_train.py`, `run_eval.py`, `evaluation/model_eval/6compare_all_evals.py`, dashboard `Model Comparison`. |
| E2 Shallow ablation | Adds shallow modifications step by step: multiscale branches, fusion, multiple heads, attention stem, bilinear fusion, and TCN blocks. | Shallow model files under `decomposers/ML_methods/NN_based/models_shallow/`, training scripts under `decomposers/ML_methods/NN_based/`, dashboard `Research Comparison`. |
| E3 Data efficiency | Repeats training with different fractions of the 8000-sample training pool. | Rerun `run_train.py` with different `--train-samples`; dashboard `Training Cost and Data Availability`. |
| E4 Real EEG / real-data check | Applies trained synthetic models to real signals and reports proxy reconstruction/separation behavior. | Dashboard `Real Data Reconstruction`, `app/services/real_data_service.py`, `evaluation/chbmit_eeg_test.py`, local `RealData/`. |
| E5 Failure mode analysis | Finds and categorizes hard signals such as weak components, overlapping sources, amplitude imbalance, and non-stationarity. | Dashboard `Test Signal`, `Error Analysis`, `app/services/statistical_hardcase_service.py`, `app/services/test_signal_diagnostics.py`. |

## Data-Efficiency Runs

The paper uses fractions of the 8000-sample training pool:

| Fraction | Training samples |
| --- | ---: |
| 5% | 400 |
| 10% | 800 |
| 25% | 2000 |
| 50% | 4000 |
| 75% | 6000 |
| 100% | 8000 |

Example PowerShell loop for the best shallow model:

```powershell
foreach ($n in 400,800,2000,4000,6000,8000) {
    py decomposers/ML_methods/NN_based/run_train.py attention_stem_multi_head_multiscale_tcn --train-samples $n --val-samples 400 --test-samples 400 --epochs 20 --seed 42
}
```

Use distinct output names or move saved model folders between runs if you want to keep every fraction side by side. Otherwise, newer runs can overwrite the same model's latest checkpoint/report files.

## Where The Paper Parameters Live

### Dataset And Signal Parameters

| Paper parameter | Paper value | Code location / argument |
| --- | --- | --- |
| Signal length | 1024 samples | Training scripts: `--signal-length`; evaluation: `evaluation/model_eval/run_eval.py --signal-length`; dataset code in `decomposers/ML_methods/NN_based/datasets/synthetic.py`. |
| Sampling rate | 256 Hz | Training scripts: `--fs`; evaluation: `run_eval.py --fs`. |
| Signal duration | 4 seconds | Implied by 1024 samples at 256 Hz. |
| Baseline components | harmonic, AM-FM, chirp | Synthetic generation code under `generators/` and app signal services under `app/services/signal_service.py`. |
| Extended components | trend, transient | Same generator/service layer as the baseline components. |
| Full training pool | 8000 samples | Pass `--train-samples 8000`. Many scripts default to 4000 for faster local runs. |
| Validation set | 400 samples | Training scripts: `--val-samples 400`. |
| Test set | 400 samples for paper-scale runs | Training scripts: `--test-samples 400`; evaluation runner: `--test-samples 400`. |
| Additive Gaussian noise | sigma sampled from [0, 0.05] | Training config fields `noise_min` and `noise_max`; CLI args `--noise-min` and `--noise-max` where exposed. |
| Random component weights | enabled | Training config `use_weights=True`; disable with `--no-weights` where supported. |
| Output component count | 3 baseline, 5 extended | Model output channels and dataset component list; training argument `--out-channels` where exposed. |
| Permutation-invariant matching | enabled | `evaluation/decomposition.py`; training/eval config `permutation_invariant_eval=True`; eval flag `--disable-permutation-invariant`. |

### Training Parameters

| Paper parameter | Paper value | Code location / argument |
| --- | --- | --- |
| Optimizer | Adam | Shared training pipeline in `decomposers/ML_methods/NN_based/training_common.py`. |
| Learning rate | 1e-3 | Training scripts: `--learning-rate 0.001`. |
| Weight decay | 1e-5 | Training scripts: `--weight-decay 0.00001`. |
| Epochs | 20 | Training scripts: `--epochs 20`. |
| Main batch size | 32 | Training scripts: `--batch-size 32`. |
| TasNet batch size | 16 | `trainTasnet.py`; pass `--batch-size 16`. |
| SepFormer batch size | 8 | `trainSepformer.py`; pass `--batch-size 8`. |
| Random seed | 42 | Training scripts: `--seed 42`. |
| Best checkpoint | lowest validation loss | Shared save logic in `training_common.py`; outputs go to each model's `saved_models` folder. |
| Component loss weight | 1.0 | Training config `component_loss_weight`; CLI arg where exposed. |
| Reconstruction loss weight | 0.5 | Training config `reconstruction_loss_weight`; CLI arg where exposed. |
| Spectral loss weight | 0.05 | Training config `spectral_loss_weight`; CLI arg where exposed. |

The combined loss used in the paper is:

```text
loss = 1.0 * component_loss + 0.5 * reconstruction_loss + 0.05 * spectral_loss
```

Loss construction and evaluation helpers are in:

```text
decomposers/ML_methods/NN_based/training_common.py
```

### Metric Parameters

| Metric family | Code location |
| --- | --- |
| Macro correlation | `evaluation/metrics.py`, `evaluation/decomposition.py`, model eval reports. |
| Macro SNR in dB | `evaluation/metrics.py`, `evaluation/decomposition.py`, model eval reports. |
| Test loss | Saved by the training/evaluation reports in each model output folder. |
| Parameter count | Calculated during training/evaluation and stored in model reports. |
| Inference time | Inference utilities in `app/services/inference_service.py` and evaluation reports. |
| Training time | Training pipeline timing in `training_common.py`; saved in report JSON files. |

## Architecture Parameters

| Model / family | Main files | Important parameters from the paper |
| --- | --- | --- |
| TasNet | `decomposers/ML_methods/NN_based/models/tasnet/`, `trainTasnet.py` | encoder kernel 16, stride 8, bottleneck 128, hidden 256, skip 128, TCN blocks/repeats, dilation growth. |
| U-Net 1D | `decomposers/ML_methods/NN_based/models/unet1d/`, `trainUnet1d.py` | base channels 32, four-level encoder/decoder structure. |
| SepFormer | `decomposers/ML_methods/NN_based/models/sepformer/`, `trainSepformer.py` | encoder kernel 16, stride 8, model/bottleneck dim 128, 8 attention heads, feedforward dim 256, chunk/hop settings. |
| RNN baseline | `decomposers/ML_methods/NN_based/models/rnnbased/`, `trainRNNbased.py` | bidirectional recurrent model, hidden size 128, two layers, LSTM/GRU option. |
| Autoencoder shallow baseline | `decomposers/ML_methods/NN_based/models_shallow/autoencoderbased/`, `trainAutoencoderBased.py` | channel widths 16, 32, 64; kernel size 5; dropout 0.1. |
| Single-head MLP | `decomposers/ML_methods/NN_based/models_shallow/mlp_singlehead/`, `trainMLPsingleHead.py` | flatten input, small hidden layer, shared output. |
| Classic Conv1D | `decomposers/ML_methods/NN_based/models_shallow/conv1Dnetwork/`, `trainConv1Dnetwork.py` | Conv1D hidden channels, kernel size 5, shared output head. The paper's simplest Stage 1 is a single Conv1D baseline; check or pass the layer-count argument if exact Stage 1 reproduction is needed. |
| Multiscale branches | `models_shallow/multiscale_branches/`, related train scripts | branch kernels 5, 7, 9; parallel shallow filters. |
| Multiscale dilated | `models_shallow/multiscale_branches/`, related train scripts | dilations 1, 2, 4 added to multiscale branches. |
| Multiple-head variants | `models_shallow/multiscale_branches/`, related train scripts | separate output heads per component. |
| Attention-stem variants | `models_shallow/multiscale_branches/`, related train scripts | convolutional stem with attention weighting, stem kernel 9. |
| Best shallow TCN | `models_shallow/multiscale_branches/attention_stem_multi_head_multiscale_tcn_decomposer.py`, `trainAttentionStemMultiHeadMultiScaleTCN.py` | branch channels 16, fused channels 48, TCN channels 48, kernels 5/7/9, branch dilations 1/2/4, TCN dilations 1/2/4/8, GroupNorm, GELU, dropout 0.1. |
| Inference-optimized shallow TCN | `trainAttentionStemMultiHeadMultiScaleTCNInferenceOptimized.py` and matching model file | smaller branch/fusion/TCN widths, shorter dilation stack, ReLU/BatchNorm-oriented settings. |

## Model Keys

Common model keys used by the launchers:

```text
unet1d
rnnbased
autoencoderbased
conv1dnetwork
mlp_singlehead
tasnet
sepformer
fuse_shallow
multiscale_dilated
multiscale_branches
multiple_head_multiscale_branches
attention_stem_multiple_head_multiscale_branches
attention_stem_bilinear_fusion_multiple_head_multiscale_branches
attention_stem_multi_head_multiscale_tcn
attention_stem_multi_head_multiscale_tcn_inference_optimized
```

If a key fails in `run_eval.py`, check whether it has an evaluation script configured in `experiment_catalog.py`. Some variants are present for training, dashboard inspection, or ablation work but are not enabled in every evaluation path.

## Outputs

Training outputs are stored under model-specific folders, usually below:

```text
decomposers/ML_methods/NN_based/saved_models/<model_name>/
```

Common files include:

```text
*_best.pt
final_test_report.json
training_history.json
latest_metrics.json
train_config.json
```

Comparison outputs are stored under:

```text
evaluation/model_eval/artifacts/
```

Dashboard exports are stored under:

```text
app/exports/
```

## Dashboard Pages

The dashboard entry point is:

```text
app/main.py
```

Useful pages for the paper:

| Dashboard page | Use |
| --- | --- |
| Model Comparison | Compare trained models, metrics, reports, and plots. |
| Research Comparison | Inspect ablation/research comparison charts. |
| Training Cost and Data Availability | Explore data-fraction and training-cost behavior. |
| Reconstruction Inspector | Run model inference on generated or selected signals. |
| Real Data Reconstruction | Apply trained models to local real signals. |
| Test Signal | Generate controlled hard cases. |
| Error Analysis | Inspect residuals and failure behavior. |
| Export Assets | Export figures/tables for the paper or presentation. |

## Reproducibility Notes

The paper was run on an NVIDIA RTX 4080M laptop GPU with an Intel i9-13900HX CPU. Exact wall-clock training time and inference latency can change with hardware, CUDA/cuDNN versions, PyTorch version, and background system load.

Use explicit command-line values when reproducing paper numbers. In particular, pass `--train-samples 8000` for full-pool experiments because several scripts use smaller defaults for day-to-day development.

The RNN baseline is known to be less stable than the convolutional and separator models. This is why the paper discusses it separately from the main stable benchmark.

## Documentation

For a concise source-code map, see:

```text
docs/FILE_GUIDE.md
```
