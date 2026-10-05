# Synthetic Fake News Corpus Ablation

This archive contains the synthetic corpora, classifier results, and analysis
outputs for a study on stylistic feature ablations in LLM-based fake news
augmentation.

## Contents

reports/final_report.md
    Aggregated results across all experiments.

results/
    metadata.json              run configuration and dataset split hashes
    baselines.json             no-augmentation, oversampling, class-weighted results
    stats.json                 per-feature ablation statistics
    token_cost.json            prompt token counts
    stratified.json            per-source evaluation
    classical/                 per-condition results for six classical classifiers
    transformers/              per-condition transformer results

generated/
    Synthetic corpora. Each file is named `{condition}__corpus{n}.json` with
    a single key `headlines` containing a list of strings.

plots/
    Figures referenced in the paper.

dataset/
    headlines.csv              GossipCop + PolitiFact headlines
    metadata.json              dataset statistics and split hashes

## Generation parameters

Models:    Qwen2.5-7B-Instruct (primary)
           Phi-3.5-mini-instruct (cross-model validation)
           DeepSeek-R1-Distill-Llama-8B (cross-model validation)
Framework: vLLM 0.11.2, bfloat16
Sampling:  temperature 0.6, top-p 0.8
Length:    60 new tokens per headline (800 for DeepSeek-R1)

## Classifier parameters

Random Forest:       200 trees, TF-IDF (5000 features, 1-2 grams)
XGBoost:             200 estimators
CatBoost:            100 iterations
LightGBM:            200 estimators
Logistic Regression: max 2000 iterations, balanced class weights
Transformers:        DistilBERT, RoBERTa, DeBERTa-v3, 2 epochs, LR 2e-5

## Seeds

Split seed: 42
Classifier seeds: 30 per corpus
Transformer seeds: 3 per corpus
Corpora per condition: 3

## Reproducibility

### Environment

All experiments were run on a single NVIDIA RTX 5090 GPU (32 GB VRAM).
The software environment at experiment time was:

- Python 3.13
- PyTorch 2.11.0+cu128 (compiled against CUDA 12.8)
- vLLM 0.11.2
- transformers 5.16.1
- scikit-learn, XGBoost, CatBoost, LightGBM, statsmodels, VADER sentiment (see requirements.txt)
- NVIDIA driver 595.84

Later versions of these libraries are not guaranteed to produce numerically
identical results. The archived deposit contains per-seed classifier
predictions, so the paper's reported numbers can be verified without
re-running the pipeline.

### Seeds

- Dataset split: random_state=42
- Classical classifier seeds: 0–29 per corpus
- Transformer seeds: 0–2 per corpus
- Bootstrap: random_state=0, 10,000 resamples

### Statistical methods

- Paired t-test on 3 corpus-level F1 means
- Benjamini–Hochberg FDR correction (q = 0.05)
- TOST equivalence at δ = 0.02
- 95% confidence intervals from the paired t-test at the corpus level (n = 3)
