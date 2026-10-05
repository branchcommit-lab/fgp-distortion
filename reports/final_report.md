# Feature Ablation Experiment

Generated: 2026-10-05T02:55:37.859378

Primary generator: Qwen/Qwen2.5-7B-Instruct
Cross-LLM models: ['microsoft/Phi-3.5-mini-instruct', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B']
Corpora per condition: 3
Classifier seeds: 30

## Baselines
- no_aug: fake F1 = 0.6034 +/- 0.0029
- oversample: fake F1 = 0.6324 +/- 0.0042
- class_weight: fake F1 = 0.6452 +/- 0.0024

## Ablation

| Feature | F1 diff | p (t) | p (FDR) | TOST p | Equivalent? | 95% CI |
|---|---|---|---|---|---|---|
| F1 | 0.0038 | 0.0041 | 0.0232 | 0.0001 | Yes | [0.0027, 0.0048] |
| F2 | 0.0016 | 0.0395 | 0.1052 | 0.0002 | Yes | [0.0002, 0.0031] |
| F3 | 0.0034 | 0.0659 | 0.1235 | 0.0015 | Yes | [-0.0005, 0.0073] |
| F4 | 0.0033 | 0.0491 | 0.1052 | 0.0010 | Yes | [0.0000, 0.0065] |
| F5 | 0.0007 | 0.2817 | 0.3521 | 0.0003 | Yes | [-0.0013, 0.0026] |
| F6 | -0.0006 | 0.5461 | 0.5851 | 0.0010 | Yes | [-0.0043, 0.0031] |
| F7 | -0.0025 | 0.0046 | 0.0232 | 0.0000 | Yes | [-0.0032, -0.0018] |
| F8 | -0.0018 | 0.3680 | 0.4246 | 0.0038 | Yes | [-0.0087, 0.0050] |
| F9 | -0.0031 | 0.0035 | 0.0232 | 0.0001 | Yes | [-0.0039, -0.0023] |
| F10 | 0.0005 | 0.2288 | 0.3120 | 0.0001 | Yes | [-0.0008, 0.0019] |
| F11 | -0.0027 | 0.0995 | 0.1658 | 0.0015 | Yes | [-0.0068, 0.0013] |
| F12 | -0.0035 | 0.0167 | 0.0626 | 0.0004 | Yes | [-0.0054, -0.0015] |
| F13 | -0.0034 | 0.0488 | 0.1052 | 0.0011 | Yes | [-0.0068, -0.0000] |
| F14 | -0.0023 | 0.1664 | 0.2496 | 0.0018 | Yes | [-0.0069, 0.0023] |
| F15 | 0.0008 | 0.7057 | 0.7057 | 0.0042 | Yes | [-0.0068, 0.0084] |

## Non-inferiority (full vs F5-only)
- F1 difference = -0.0080, p = 0.0111, margin = 0.02
- Non-inferior: True

**F5 is sufficient (F5-only prompt is non-inferior to full) but not proven necessary (removing F5 does not significantly hurt).**

## Compliance table
| Feature | Compliance | Band |
|---|---|---|
| F1 |  41.4% | [0.1, 0.3] |
| F2 |   0.0% | [0.13, 0.35] |
| F3 |   5.4% | [0.02, 0.1] |
| F4 |  51.5% | [0.1, 0.8] |
| F5 |   2.1% | [1.0, 2.0] |
| F6 |  98.0% | [0.0, 0.05] |
| F7 |  27.3% | [10.0, 11.0] |
| F8 |  55.7% | [0.05, 0.35] |
| F9 |  14.1% | [0.01, 0.1] |
| F10 |   0.3% | [0.2, 0.6] |
| F11 |  11.9% | [6.0, 8.0] |
| F12 |  35.1% | [0.05, 0.3] |
| F13 |  55.3% | [0.02, 0.2] |
| F14 |  56.5% | [0.05, 0.35] |
| F15 |   0.0% | [0.13, 0.35] |
| Average | 30.3% | - |

## Target vs synthetic distribution
| Feature | Real Fake | Full | Naive | F5-only | F6-only | F12-only |
|---|---|---|---|---|---|---|
| excl | 0.071 | 0.030 | 0.006 | 1.060 | 0.010 | 0.060 |
| quest | 0.125 | 0.670 | 0.001 | 0.011 | 0.002 | 0.027 |
| caps | 0.247 | 0.019 | 0.092 | 0.058 | 7.358 | 0.004 |
| words | 11.106 | 8.766 | 10.103 | 8.273 | 8.720 | 5.568 |

| Feature | Full | Naive | F5-only | F6-only | F12-only |
|---|---|---|---|---|---|
| excl | 0.42x | 0.08x | 14.91x | 0.15x | 0.84x |
| quest | 5.34x | 0.01x | 0.09x | 0.01x | 0.21x |
| caps | 0.08x | 0.37x | 0.24x | 29.74x | 0.02x |
| words | 0.79x | 0.91x | 0.74x | 0.79x | 0.50x |

## Cross-family compliance ratios
| Model | Feature | Full | Naive | F5-only |
|---|---|---|---|---|
| Phi-3.5-mini-instruct | excl | 11.28x | 0.15x | 15.72x |
| Phi-3.5-mini-instruct | quest | 4.51x | 0.07x | 0.23x |
| Phi-3.5-mini-instruct | caps | 0.10x | 0.11x | 0.12x |
| Phi-3.5-mini-instruct | words | 0.89x | 0.75x | 0.84x |
| DeepSeek-R1-Distill-Llama-8B | excl | 1.63x | 0.50x | 5.92x |
| DeepSeek-R1-Distill-Llama-8B | quest | 0.66x | 0.19x | 0.05x |
| DeepSeek-R1-Distill-Llama-8B | caps | 0.10x | 0.68x | 0.13x |
| DeepSeek-R1-Distill-Llama-8B | words | 1.66x | 1.75x | 1.46x |
| Qwen2.5-7B-Instruct | excl | 0.42x | 0.08x | 14.91x |
| Qwen2.5-7B-Instruct | quest | 5.34x | 0.01x | 0.09x |
| Qwen2.5-7B-Instruct | caps | 0.08x | 0.37x | 0.24x |
| Qwen2.5-7B-Instruct | words | 0.79x | 0.91x | 0.74x |

## Transformer results
- full c0 distilbert-base-uncased: fake F1 = 0.6908 +/- 0.0031
- full c0 roberta-base: fake F1 = 0.7079 +/- 0.0029
- full c0 microsoft/deberta-v3-base: fake F1 = 0.6999 +/- 0.0018
- full c1 distilbert-base-uncased: fake F1 = 0.6925 +/- 0.0019
- full c1 roberta-base: fake F1 = 0.7079 +/- 0.0042
- full c1 microsoft/deberta-v3-base: fake F1 = 0.6995 +/- 0.0009
- full c2 distilbert-base-uncased: fake F1 = 0.6930 +/- 0.0011
- full c2 roberta-base: fake F1 = 0.7066 +/- 0.0050
- full c2 microsoft/deberta-v3-base: fake F1 = 0.6980 +/- 0.0009
- naive c0 distilbert-base-uncased: fake F1 = 0.6964 +/- 0.0032
- naive c0 roberta-base: fake F1 = 0.7114 +/- 0.0037
- naive c0 microsoft/deberta-v3-base: fake F1 = 0.7093 +/- 0.0033
- naive c1 distilbert-base-uncased: fake F1 = 0.6984 +/- 0.0016
- naive c1 roberta-base: fake F1 = 0.7076 +/- 0.0036
- naive c1 microsoft/deberta-v3-base: fake F1 = 0.7005 +/- 0.0044
- naive c2 distilbert-base-uncased: fake F1 = 0.6977 +/- 0.0033
- naive c2 roberta-base: fake F1 = 0.7127 +/- 0.0034
- naive c2 microsoft/deberta-v3-base: fake F1 = 0.7019 +/- 0.0061
- only_F5 c0 distilbert-base-uncased: fake F1 = 0.6994 +/- 0.0016
- only_F5 c0 roberta-base: fake F1 = 0.7122 +/- 0.0052
- only_F5 c0 microsoft/deberta-v3-base: fake F1 = 0.7029 +/- 0.0072
- only_F5 c1 distilbert-base-uncased: fake F1 = 0.7008 +/- 0.0029
- only_F5 c1 roberta-base: fake F1 = 0.7112 +/- 0.0051
- only_F5 c1 microsoft/deberta-v3-base: fake F1 = 0.7030 +/- 0.0072
- only_F5 c2 distilbert-base-uncased: fake F1 = 0.6995 +/- 0.0022
- only_F5 c2 roberta-base: fake F1 = 0.7094 +/- 0.0045
- only_F5 c2 microsoft/deberta-v3-base: fake F1 = 0.7038 +/- 0.0056

## LLM classifier
{
  "accuracy": 0.6866379310344828,
  "f1_fake": 0.32372093023255816,
  "precision_fake": 0.3483483483483483,
  "recall_fake": 0.3023457862728063
}

## Token table
{
  "full_tokens": 183,
  "minimal_F5_tokens": 58,
  "naive_tokens": 9,
  "reduction_pct": 68.30601092896174
}

## Stratified analysis
{
  "full": {
    "politifact": {
      "accuracy": 0.7877358490566038,
      "f1_fake": 0.7486033519553073,
      "f1_real": 0.8163265306122449,
      "macro_f1": 0.7824649412837761,
      "precision_fake": 0.7282608695652174,
      "recall_fake": 0.7701149425287356,
      "tn": 100,
      "fp": 25,
      "fn": 20,
      "tp": 67,
      "roc_auc": 0.8724597701149427
    },
    "gossipcop": {
      "accuracy": 0.8360433604336044,
      "f1_fake": 0.5975609756097561,
      "f1_real": 0.8970504821327283,
      "macro_f1": 0.7473057288712421,
      "precision_fake": 0.7293640054127198,
      "recall_fake": 0.5061032863849765,
      "tn": 3163,
      "fp": 200,
      "fn": 526,
      "tp": 539,
      "roc_auc": 0.8471837268032817
    }
  },
  "only_F5": {
    "politifact": {
      "accuracy": 0.7594339622641509,
      "f1_fake": 0.7487684729064039,
      "f1_real": 0.7692307692307693,
      "macro_f1": 0.7589996210685865,
      "precision_fake": 0.6551724137931034,
      "recall_fake": 0.8735632183908046,
      "tn": 85,
      "fp": 40,
      "fn": 11,
      "tp": 76,
      "roc_auc": 0.8802298850574714
    },
    "gossipcop": {
      "accuracy": 0.8310749774164409,
      "f1_fake": 0.6067297581493165,
      "f1_real": 0.8924360080529192,
      "macro_f1": 0.7495828831011179,
      "precision_fake": 0.6893667861409797,
      "recall_fake": 0.5417840375586854,
      "tn": 3103,
      "fp": 260,
      "fn": 488,
      "tp": 577,
      "roc_auc": 0.8414790058619135
    }
  }
}

## Cross-LLM generation
- microsoft/Phi-3.5-mini-instruct: ['full', 'naive', 'only_F5']
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B: ['full', 'naive', 'only_F5']