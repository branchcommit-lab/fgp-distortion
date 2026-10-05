import os, sys, gc, time, json, shutil, hashlib, resource
import re
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List

warnings.filterwarnings("ignore")

try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 1048576), _hard))
except Exception:
    pass

for k, v in {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_VERBOSITY": "error",
    "TQDM_DISABLE": "1",
}.items():
    os.environ[k] = v

for pkg in ["accelerate", "datasets", "scikit-learn", "xgboost",
            "catboost", "lightgbm", "vaderSentiment", "textstat",
            "sentencepiece", "statsmodels"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import numpy as np
import pandas as pd
import torch
import vllm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
from statsmodels.stats.multitest import multipletests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sns.set_style("whitegrid")

RUN_TAG = "v1"
ROOT = Path("/home/jovyan/fake_news_experiment").resolve()
CACHE = ROOT / f"cache_{RUN_TAG}"
HF_CACHE = ROOT / "hf_cache"

WIPE_EXPERIMENT_CACHE = False
WIPE_HF_CACHE = False

if WIPE_EXPERIMENT_CACHE and CACHE.exists():
    shutil.rmtree(CACHE, ignore_errors=True)
if WIPE_HF_CACHE and HF_CACHE.exists():
    shutil.rmtree(HF_CACHE, ignore_errors=True)

DATASET_CACHE = CACHE / "dataset"
GEN_CACHE = CACHE / "generated"
RESULTS = CACHE / "results"
PLOTS = CACHE / "plots"
REPORTS = CACHE / "reports"
for p in [ROOT, CACHE, DATASET_CACHE, GEN_CACHE, RESULTS, PLOTS, REPORTS, HF_CACHE]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")

MODE = "full"
SCALE = {
    "pilot": {"n_synth": 500, "n_corpora": 1, "clf_seeds": 3, "tf_seeds": 1},
    "full":  {"n_synth": 5000, "n_corpora": 3, "clf_seeds": 30, "tf_seeds": 3},
}[MODE]


@dataclass
class Config:
    n_synthetic_per_corpus: int = SCALE["n_synth"]
    n_corpora: int = SCALE["n_corpora"]
    classifier_seeds: int = SCALE["clf_seeds"]
    transformer_seeds: int = SCALE["tf_seeds"]
    vllm_prompts_per_batch: int = 128
    headlines_per_prompt: int = 4
    max_new_tokens: int = 60
    temperature: float = 0.6
    top_p: float = 0.8
    gpu_memory_utilization: float = 0.85
    test_size: float = 0.2
    split_seed: int = 42
    noninf_margin: float = 0.02
    equivalence_margin: float = 0.02
    n_bootstrap: int = 10000
    generator_models: List[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ])
    llm_classifier_model: str = "Qwen/Qwen2.5-7B-Instruct"
    transformer_classifiers: List[str] = field(default_factory=lambda: [
        "distilbert-base-uncased",
        "roberta-base",
        "microsoft/deberta-v3-base",
    ])
    transformer_batch_size: int = 64
    transformer_max_len: int = 64
    transformer_epochs: int = 2
    num_workers: int = 0


CFG = Config()

FEATURES = [
    ("F1",  "Speculation ratio",        "Use speculative words (maybe, could, allegedly) in ~15-25% of sentences."),
    ("F2",  "Question marks",           "Include a question in 13-35% of headlines. Use '?'."),
    ("F3",  "Attribution density",      "Use vague authority references (officials say, experts warn) at low density (~6.5%)."),
    ("F4",  "Emotional intensity",      "Use emotionally charged words (fear, anger, urgency)."),
    ("F5",  "Exclamation marks",        "Include 1-2 exclamation marks per headline."),
    ("F6",  "ALL CAPS words",           "Use all-caps words in <5% of words."),
    ("F7",  "Word count",               "Keep headline length 10-11 words."),
    ("F8",  "Quotation marks",          "Include direct or scare quotes occasionally."),
    ("F9",  "Numerical references",     "Mention numbers, statistics, or percentages."),
    ("F10", "Repetition ratio",         "Repeat some words or phrases slightly more than real news."),
    ("F11", "Readability",              "Keep reading grade level slightly lower than real news (e.g., Flesch-Kincaid grade 6-8)."),
    ("F12", "First-person and inclusive pronouns", "Use 'we', 'you', 'people' more often."),
    ("F13", "Hedging language",         "Use 'might', 'possibly', 'suggests'."),
    ("F14", "Authority misattribution", "Refer to anonymous insiders or 'sources say'."),
    ("F15", "Question framing",         "Phrase the headline as a question 13-35% of the time."),
]
FEATURE_IDS = [f[0] for f in FEATURES]
FEATURE_DESC = {f[0]: f[2] for f in FEATURES}

INTERACTION_COMBOS = [
    ["F5", "F6"], ["F5", "F4"], ["F6", "F4"],
    ["F5", "F6", "F4"], ["F5", "F12"], ["F5", "F6", "F12"],
]

SINGLE_FEATURE_SUFFICIENCY = ["F5", "F6", "F12"]

SPEC_WORDS = {"maybe", "could", "might", "allegedly", "perhaps", "possibly",
              "supposedly", "rumor", "reportedly"}
HEDGE_WORDS = {"might", "possibly", "suggests", "suggest", "may", "could",
               "appears", "seems", "likely", "unlikely"}
FIRST_PERSON = {"i", "we", "you", "our", "us", "people", "my", "your"}
ATTRIBUTION_WORDS = {"officials", "experts", "sources", "insiders",
                     "authorities", "reports", "spokesperson"}
AUTHORITY_PHRASES = [
    "sources say", "insiders say", "anonymous source",
    "sources claim", "insiders reveal", "a source said",
    "unnamed official", "anonymous insider",
    "anonymous official", "according to insiders",
]

BANDS = {
    "F1": (0.10, 0.30), "F2": (0.13, 0.35), "F3": (0.02, 0.10),
    "F4": (0.10, 0.80), "F5": (1.0, 2.0), "F6": (0.00, 0.05),
    "F7": (10.0, 11.0), "F8": (0.05, 0.35), "F9": (0.01, 0.10),
    "F10": (0.20, 0.60), "F11": (6.0, 8.0), "F12": (0.05, 0.30),
    "F13": (0.02, 0.20), "F14": (0.05, 0.35), "F15": (0.13, 0.35),
}
RATE_BASED = {"F2", "F5", "F8", "F14", "F15"}

_vader = SentimentIntensityAnalyzer()


def measure_feature(h, fid):
    hl = h.lower()
    toks = hl.split()
    n = max(len(toks), 1)
    if fid == "F1":
        return sum(1 for w in toks if w in SPEC_WORDS) / n
    if fid == "F2":
        return 1.0 if "?" in h else 0.0
    if fid == "F3":
        c = sum(1 for w in toks if w in ATTRIBUTION_WORDS)
        c += sum(1 for p in AUTHORITY_PHRASES if p in hl)
        return c / n
    if fid == "F4":
        return abs(_vader.polarity_scores(h)["compound"])
    if fid == "F5":
        return float(h.count("!"))
    if fid == "F6":
        if not toks:
            return 0.0
        return sum(1 for w in h.split() if w.isupper() and len(w) > 1) / n
    if fid == "F7":
        return float(n)
    if fid == "F8":
        return 1.0 if re.search(r'"[^"]+"', h) else 0.0
    if fid == "F9":
        return sum(1 for c in h if c.isdigit()) / max(len(h), 1)
    if fid == "F10":
        return 1.0 - len(set(toks)) / n if toks else 0.0
    if fid == "F11":
        try:
            return float(textstat.flesch_kincaid_grade(h))
        except Exception:
            return sum(len(w) for w in toks) / n
    if fid == "F12":
        return sum(1 for w in toks if w in FIRST_PERSON) / n
    if fid == "F13":
        return sum(1 for w in toks if w in HEDGE_WORDS) / n
    if fid == "F14":
        return 1.0 if any(p in hl for p in AUTHORITY_PHRASES) else 0.0
    if fid == "F15":
        return 1.0 if h.rstrip().endswith("?") else 0.0
    return 0.0


def band_score(value, lo, hi):
    if lo <= value <= hi:
        return 1.0
    mid = (lo + hi) / 2.0
    span = max((hi - lo) / 2.0, 1e-6)
    return max(0.0, 1.0 - abs(value - mid) / (3.0 * span))


def feature_compliance_summary(headlines):
    out = {}
    for fid in FEATURE_IDS:
        lo, hi = BANDS[fid]
        vals = [measure_feature(h, fid) for h in headlines]
        if not vals:
            out[fid] = 0.0
            continue
        if fid == "F5":
            out[fid] = band_score(float(np.mean(vals)), lo, hi)
        elif fid in RATE_BASED:
            rate = float(np.mean([1.0 if v >= 1.0 else 0.0 for v in vals]))
            out[fid] = band_score(rate, lo, hi)
        else:
            out[fid] = float(np.mean([1.0 if lo <= v <= hi else 0.0 for v in vals]))
    return out


def load_dataset():
    path = DATASET_CACHE / "headlines.csv"
    if path.exists():
        return pd.read_csv(path)
    urls = {
        "politifact_fake": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv",
        "politifact_real": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv",
        "gossipcop_fake":  "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv",
        "gossipcop_real":  "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv",
    }
    rows = []
    for name, url in urls.items():
        d = pd.read_csv(url)
        if "title" not in d.columns:
            continue
        p = d[["title"]].copy()
        p.columns = ["text"]
        p["label"] = 1 if "fake" in name else 0
        p["source"] = "politifact" if "politifact" in name else "gossipcop"
        rows.append(p)
    out = pd.concat(rows, ignore_index=True).dropna(subset=["text"]).reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


df = load_dataset()
TEXTS = df["text"].tolist()
LABELS = df["label"].tolist()
SOURCES = df["source"].tolist()

sss = StratifiedShuffleSplit(n_splits=1, test_size=CFG.test_size, random_state=CFG.split_seed)
TRAIN_IDX, TEST_IDX = next(sss.split(TEXTS, LABELS))
PER_SOURCE = {}
for src in df["source"].unique():
    sub = df[df["source"] == src]
    s = StratifiedShuffleSplit(n_splits=1, test_size=CFG.test_size, random_state=CFG.split_seed)
    a, b = next(s.split(sub["text"], sub["label"]))
    PER_SOURCE[src] = {"train_idx": sub.index.values[a], "test_idx": sub.index.values[b]}

meta = {
    "config": asdict(CFG),
    "n_samples": len(TEXTS),
    "n_fake": int(sum(LABELS)),
    "n_real": int(len(LABELS) - sum(LABELS)),
    "train_idx_hash": hashlib.md5(TRAIN_IDX.tobytes()).hexdigest(),
    "test_idx_hash": hashlib.md5(TEST_IDX.tobytes()).hexdigest(),
    "timestamp": datetime.now().isoformat(),
}
(RESULTS / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

_vllm_engines = {}


def load_engine(model_name):
    if model_name in _vllm_engines:
        return _vllm_engines[model_name]
    from vllm import LLM
    is_reasoning = "deepseek" in model_name.lower() and "r1" in model_name.lower()
    kw = dict(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=CFG.gpu_memory_utilization,
        max_model_len=2048,
        download_dir=str(HF_CACHE),
        enforce_eager=is_reasoning,
    )
    if "phi" in model_name.lower():
        kw["trust_remote_code"] = True
    engine = LLM(**kw)
    _vllm_engines[model_name] = engine
    return engine


def _wait_vram(min_ratio=0.85, timeout=30):
    try:
        for _ in range(int(timeout * 2)):
            free, total = torch.cuda.mem_get_info()
            if free / total >= min_ratio:
                return True
            time.sleep(0.5)
    except Exception:
        pass
    return False


def free_engine(model_name=None):
    names = [model_name] if model_name else list(_vllm_engines.keys())
    for n in list(names):
        if n in _vllm_engines:
            try:
                eng = _vllm_engines.pop(n)
                for attr in ("shutdown", "close", "stop"):
                    if hasattr(eng, attr):
                        try:
                            getattr(eng, attr)()
                        except Exception:
                            pass
                del eng
            except Exception:
                pass
    gc.collect()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    try:
        import psutil
        me = psutil.Process()
        my_user = me.username()
        my_pid = me.pid
        my_children = {c.pid for c in me.children(recursive=True)}
        for proc in psutil.process_iter(["pid", "cmdline", "username"]):
            try:
                pid = proc.info["pid"]
                if pid == my_pid:
                    continue
                if proc.info.get("username") != my_user:
                    continue
                cmd = " ".join(proc.info.get("cmdline") or [])
                is_engine = ("EngineCore" in cmd
                             or "vllm.v1.engine.core" in cmd
                             or ("vllm" in cmd.lower() and "worker" in cmd.lower()))
                if is_engine and (pid in my_children or "vllm" in cmd.lower()):
                    proc.kill()
            except Exception:
                pass
    except ImportError:
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    _wait_vram()


BASE_PROMPT = """Generate {n} realistic fake news headlines with the following stylistic guidelines:

{feature_constraints}

Guidelines:
- Headlines should be plausible and mimic real fake news patterns.
- Avoid extreme exaggeration or obvious non-news content.
- Vary the topics (celebrity, politics, health, etc.).
- Output one headline per line, nothing else.

Generate exactly {n} headlines:"""

NAIVE_PROMPT = "Generate {n} fake news headlines. Output one per line."

_think_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text):
    text = _think_re.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def build_prompt(active, n):
    if not active:
        return NAIVE_PROMPT.format(n=n)
    return BASE_PROMPT.format(
        feature_constraints="\n".join(f"- {FEATURE_DESC[f]}" for f in active),
        n=n,
    )


def split_headlines(text, max_n):
    text = strip_think(text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = ln.lstrip("-*0123456789. )").strip()
        if 4 < len(ln) < 200:
            cleaned.append(ln)
    return cleaned[:max_n]


def generate_corpus(model_name, condition, corpus_idx, active, n):
    d = GEN_CACHE / model_name.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{condition}__corpus{corpus_idx}.json"

    if f.exists():
        try:
            data = json.loads(f.read_text())
            if isinstance(data.get("headlines"), list) and data["headlines"]:
                return data["headlines"]
            f.unlink()
        except Exception:
            f.unlink()

    print(f"generate {model_name} {condition} c{corpus_idx}")
    engine = load_engine(model_name)
    from vllm import SamplingParams
    prompt_tmpl = build_prompt(active, n)
    k = CFG.headlines_per_prompt
    max_tok = CFG.max_new_tokens * k
    if "deepseek" in model_name.lower() and "r1" in model_name.lower():
        max_tok = max(max_tok, 800)

    headlines = []
    batch = CFG.vllm_prompts_per_batch
    while len(headlines) < n:
        remaining = n - len(headlines)
        n_prompts = min(batch, (remaining + k - 1) // k)
        prompts = [prompt_tmpl.replace("{n}", str(k))] * n_prompts
        sp = SamplingParams(max_tokens=max_tok,
                            temperature=max(CFG.temperature, 0.01),
                            top_p=CFG.top_p, n=1)
        try:
            outs = engine.generate(prompts, sp)
        except Exception as e:
            print("generation error", e)
            break
        for o in outs:
            headlines += split_headlines(o.outputs[0].text, k)
    headlines = headlines[:n]
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"headlines": headlines}, indent=0))
    tmp.replace(f)
    print(f"  {condition} c{corpus_idx}: {len(headlines)} headlines")
    return headlines


def get_clf(name, seed):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=200, random_state=seed,
                                  eval_metric="logloss", verbosity=0, n_jobs=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=100, random_seed=seed,
                                   verbose=False, thread_count=-1)
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, random_state=seed,
                                   n_jobs=-1, verbose=-1)
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
    if name == "voting":
        import lightgbm as lgb
        base = [
            ("rf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
            ("lgb", lgb.LGBMClassifier(n_estimators=200, random_state=seed, n_jobs=-1, verbose=-1)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed, n_jobs=-1)),
        ]
        return VotingClassifier(estimators=base, voting="soft")
    raise ValueError(name)


_tfidf_cache = {}


def cached_tfidf(real_texts, train_idx, test_idx, synth,
                 max_features=5000, ngram_range=(1, 2)):
    h = hashlib.md5()
    h.update(str(len(train_idx)).encode())
    h.update(str(len(synth)).encode())
    for i in train_idx[:50]:
        h.update(real_texts[i].encode(errors="ignore")[:50])
    for s in synth[:50]:
        h.update(s.encode(errors="ignore")[:50])
    key = h.hexdigest()
    if key in _tfidf_cache:
        return _tfidf_cache[key]
    x_text = [real_texts[i] for i in train_idx] + list(synth)
    y_text = [real_texts[i] for i in test_idx]
    v = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                        lowercase=True, strip_accents="unicode")
    Xtr = v.fit_transform(x_text)
    Xte = v.transform(y_text)
    _tfidf_cache[key] = (Xtr, Xte)
    return Xtr, Xte


def eval_classical(name, seed, real_texts, real_labels, train_idx, test_idx, synth):
    Xtr, Xte = cached_tfidf(real_texts, train_idx, test_idx, synth)
    ytr = [real_labels[i] for i in train_idx] + [1] * len(synth)
    yte = [real_labels[i] for i in test_idx]
    clf = get_clf(name, seed).fit(Xtr, ytr)
    p = clf.predict(Xte)
    pr = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else None
    tn, fp, fn, tp = confusion_matrix(yte, p, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(yte, p),
        "f1_fake": f1_score(yte, p, pos_label=1),
        "f1_real": f1_score(yte, p, pos_label=0),
        "macro_f1": f1_score(yte, p, average="macro"),
        "precision_fake": precision_score(yte, p, pos_label=1, zero_division=0),
        "recall_fake": recall_score(yte, p, pos_label=1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if pr is not None:
        try:
            out["roc_auc"] = roc_auc_score(yte, pr)
        except Exception:
            out["roc_auc"] = float("nan")
    return out


def run_classical(condition, corpus_idx, clf_name, synth, model_name):
    safe = model_name.replace("/", "__")
    f = RESULTS / "classical" / f"{safe}__{condition}__c{corpus_idx}__{clf_name}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return json.loads(f.read_text())
    runs = [eval_classical(clf_name, s, TEXTS, LABELS, TRAIN_IDX, TEST_IDX, synth)
            for s in range(CFG.classifier_seeds)]
    agg = {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs]))} for k in runs[0]}
    out = {"aggregated": agg, "per_seed": runs}
    f.write_text(json.dumps(out, indent=2))
    return out


def get_tokenizer(model_name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name, cache_dir=str(HF_CACHE), use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, cache_dir=str(HF_CACHE), use_fast=False)


def _warmup(opt, warmup_steps, total_steps):
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = max(int(warmup_steps), 1)
    total_steps = max(int(total_steps), warmup_steps + 1)
    def fn(s):
        if s < warmup_steps:
            return s / warmup_steps
        return max(0.0, (total_steps - s) / (total_steps - warmup_steps))
    return LambdaLR(opt, fn)


def eval_transformer(model_name, seed, real_texts, real_labels,
                     train_idx, test_idx, synth, epochs=None):
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification
    epochs = epochs or CFG.transformer_epochs
    device = torch.device("cuda")
    tok = get_tokenizer(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.sep_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir=str(HF_CACHE)).to(device)

    Xtr = [real_texts[i] for i in train_idx] + list(synth)
    ytr = [real_labels[i] for i in train_idx] + [1] * len(synth)
    Xte = [real_texts[i] for i in test_idx]
    yte = [real_labels[i] for i in test_idx]

    class DS(Dataset):
        def __init__(self, xs, ys):
            self.xs, self.ys = xs, ys
        def __len__(self):
            return len(self.xs)
        def __getitem__(self, i):
            e = tok(self.xs[i], truncation=True, padding="max_length",
                    max_length=CFG.transformer_max_len, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(self.ys[i])

    torch.manual_seed(seed)
    tl = DataLoader(DS(Xtr, ytr), batch_size=CFG.transformer_batch_size,
                    shuffle=True, num_workers=CFG.num_workers)
    vl = DataLoader(DS(Xte, yte), batch_size=CFG.transformer_batch_size * 2,
                    num_workers=CFG.num_workers)

    opt = AdamW(model.parameters(), lr=2e-5)
    total = len(tl) * epochs
    sch = _warmup(opt, 0.1 * total, total)

    model.train()
    for _ in range(epochs):
        for b, y in tl:
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**b, labels=y)
            out.loss.backward()
            opt.step()
            sch.step()

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for b, _ in vl:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**b).logits
            probs += torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
            preds += logits.argmax(-1).cpu().tolist()

    tn, fp, fn, tp = confusion_matrix(yte, preds, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(yte, preds),
        "f1_fake": f1_score(yte, preds, pos_label=1),
        "precision_fake": precision_score(yte, preds, pos_label=1, zero_division=0),
        "recall_fake": recall_score(yte, preds, pos_label=1),
        "roc_auc": roc_auc_score(yte, probs),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    del model
    torch.cuda.empty_cache()
    return out


def run_transformer(condition, c_idx, model_name, synth):
    safe = model_name.replace("/", "__")
    f = RESULTS / "transformers" / f"{condition}__c{c_idx}__{safe}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return json.loads(f.read_text())
    runs = [eval_transformer(model_name, s, TEXTS, LABELS, TRAIN_IDX, TEST_IDX, synth)
            for s in range(CFG.transformer_seeds)]
    agg = {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs]))} for k in runs[0]}
    out = {"aggregated": agg, "per_seed": runs}
    f.write_text(json.dumps(out, indent=2))
    return out


LLM_CLS = "Classify this headline as FAKE or REAL. Answer one word only.\n\nHeadline: {h}\nAnswer:"


def run_llm_classifier(model_name):
    safe = model_name.replace("/", "__")
    f = RESULTS / f"llm_classifier__{safe}.json"
    if f.exists():
        return json.loads(f.read_text())
    engine = load_engine(model_name)
    from vllm import SamplingParams
    te_texts = [TEXTS[i] for i in TEST_IDX]
    te_labels = [LABELS[i] for i in TEST_IDX]
    preds = []
    batch = 128
    for i in range(0, len(te_texts), batch):
        chunk = [LLM_CLS.format(h=h) for h in te_texts[i:i + batch]]
        sp = SamplingParams(max_tokens=4, temperature=0.0, n=1)
        outs = engine.generate(chunk, sp)
        for o in outs:
            t = o.outputs[0].text.upper()
            preds.append(1 if "FAKE" in t else 0)
    out = {
        "accuracy": accuracy_score(te_labels, preds),
        "f1_fake": f1_score(te_labels, preds, pos_label=1),
        "precision_fake": precision_score(te_labels, preds, pos_label=1, zero_division=0),
        "recall_fake": recall_score(te_labels, preds, pos_label=1),
    }
    f.write_text(json.dumps(out, indent=2))
    return out


def _no_aug(seed):
    Xtr = [TEXTS[i] for i in TRAIN_IDX]
    ytr = [LABELS[i] for i in TRAIN_IDX]
    Xte = [TEXTS[i] for i in TEST_IDX]
    yte = [LABELS[i] for i in TEST_IDX]
    v = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr, Xte = v.fit_transform(Xtr), v.transform(Xte)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return {"f1_fake": f1_score(yte, p, pos_label=1),
            "precision_fake": precision_score(yte, p, pos_label=1, zero_division=0),
            "recall_fake": recall_score(yte, p, pos_label=1),
            "accuracy": accuracy_score(yte, p)}


def _oversample(seed):
    Xtr = [TEXTS[i] for i in TRAIN_IDX]
    ytr = [LABELS[i] for i in TRAIN_IDX]
    fake = [i for i, y in enumerate(ytr) if y == 1]
    real = [i for i, y in enumerate(ytr) if y == 0]
    rng = np.random.RandomState(seed)
    extra = rng.choice(fake, size=max(0, len(real) - len(fake)), replace=True)
    Xtr = Xtr + [Xtr[i] for i in extra]
    ytr = ytr + [1] * len(extra)
    Xte = [TEXTS[i] for i in TEST_IDX]
    yte = [LABELS[i] for i in TEST_IDX]
    v = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr, Xte = v.fit_transform(Xtr), v.transform(Xte)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return {"f1_fake": f1_score(yte, p, pos_label=1),
            "precision_fake": precision_score(yte, p, pos_label=1, zero_division=0),
            "recall_fake": recall_score(yte, p, pos_label=1),
            "accuracy": accuracy_score(yte, p)}


def _class_weight(seed):
    Xtr = [TEXTS[i] for i in TRAIN_IDX]
    ytr = [LABELS[i] for i in TRAIN_IDX]
    Xte = [TEXTS[i] for i in TEST_IDX]
    yte = [LABELS[i] for i in TEST_IDX]
    v = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr, Xte = v.fit_transform(Xtr), v.transform(Xte)
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    w = {int(c): float(ww) for c, ww in zip(np.unique(ytr), cw)}
    clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                  class_weight=w, n_jobs=-1).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return {"f1_fake": f1_score(yte, p, pos_label=1),
            "precision_fake": precision_score(yte, p, pos_label=1, zero_division=0),
            "recall_fake": recall_score(yte, p, pos_label=1),
            "accuracy": accuracy_score(yte, p)}


def run_baselines():
    f = RESULTS / "baselines.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for name, fn in [("no_aug", _no_aug), ("oversample", _oversample),
                     ("class_weight", _class_weight)]:
        runs = [fn(s) for s in range(CFG.classifier_seeds)]
        out[name] = {k: {"mean": float(np.mean([r[k] for r in runs])),
                          "std": float(np.std([r[k] for r in runs]))} for k in runs[0]}
    f.write_text(json.dumps(out, indent=2))
    return out


def paired_ttest(d):
    if len(d) < 2:
        return {"t": float("nan"), "p": float("nan"), "n": len(d)}
    t, p = stats.ttest_1samp(d, 0.0)
    return {"t": float(t), "p": float(p), "n": int(len(d)),
            "mean": float(np.mean(d)), "std": float(np.std(d, ddof=1))}


def tost(d, margin, alpha=0.05):
    n = len(d)
    if n < 2:
        return {"p_equiv": float("nan"), "mean": float(np.mean(d)) if n else float("nan")}
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    df = n - 1
    p_l = 1 - stats.t.cdf((m + margin) / se, df)
    p_u = stats.t.cdf((m - margin) / se, df)
    p_eq = max(p_l, p_u)
    tc = stats.t.ppf(1 - alpha, df)
    return {"p_equiv": float(p_eq), "mean": m, "se": se,
            "ci_low": float(m - tc * se), "ci_high": float(m + tc * se),
            "equiv_at_alpha": bool(p_eq < alpha)}


def t_ci(deltas, alpha=0.05):
    n = len(deltas)
    m = float(np.mean(deltas))
    if n < 2:
        return m, float("nan"), float("nan")
    sd = float(np.std(deltas, ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    return m, m - tc * se, m + tc * se


def fdr(pvals, q=0.05):
    if not pvals:
        return [], []
    rej, padj, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    return list(padj), list(rej)


def noninf(d, margin, alpha=0.05):
    n = len(d)
    if n < 2:
        return {"p": float("nan"), "mean": float(np.mean(d)) if n else float("nan")}
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    t = (m - margin) / se
    p = stats.t.cdf(t, df=n - 1)
    return {"p": float(p), "mean": m, "se": se,
            "noninferior_at_alpha": bool(p < alpha)}


def all_conditions():
    c = {"full": FEATURE_IDS[:], "naive": []}
    for fid in FEATURE_IDS:
        c[f"ablate_{fid}"] = [f for f in FEATURE_IDS if f != fid]
    for fid in SINGLE_FEATURE_SUFFICIENCY:
        c[f"only_{fid}"] = [fid]
    for combo in INTERACTION_COMBOS:
        name = "only_" + "_".join(combo)
        if name not in c:
            c[name] = list(combo)
    return c


CONDITIONS = all_conditions()
PRIMARY_MODEL = CFG.generator_models[0]
CROSS_LLM_MODELS = CFG.generator_models[1:]
CROSS_LLM_CONDS = ["full", "naive", "only_F5"]


def stage_generation():
    print("stage: generation")
    try:
        for cond, feats in CONDITIONS.items():
            for c in range(CFG.n_corpora):
                generate_corpus(PRIMARY_MODEL, cond, c, feats,
                                CFG.n_synthetic_per_corpus)
    finally:
        free_engine(PRIMARY_MODEL)
    for m in CROSS_LLM_MODELS:
        try:
            for cond in CROSS_LLM_CONDS:
                for c in range(CFG.n_corpora):
                    generate_corpus(m, cond, c, CONDITIONS[cond],
                                    CFG.n_synthetic_per_corpus)
        finally:
            free_engine(m)


def stage_llm_classifier():
    print("stage: llm classifier")
    try:
        result = run_llm_classifier(CFG.llm_classifier_model)
        print(json.dumps(result, indent=2))
    finally:
        free_engine(CFG.llm_classifier_model)


def stage_classical():
    print("stage: classical classifiers")
    names = ["rf", "xgboost", "catboost", "lightgbm", "logreg", "voting"]

    def load_corpus(m, cond, c):
        f = GEN_CACHE / m.replace("/", "__") / f"{cond}__corpus{c}.json"
        return json.loads(f.read_text())["headlines"] if f.exists() else []

    for cond in CONDITIONS:
        for c in range(CFG.n_corpora):
            corpus = load_corpus(PRIMARY_MODEL, cond, c)
            if not corpus:
                continue
            for clf in names:
                run_classical(cond, c, clf, corpus, PRIMARY_MODEL)
    for m in CROSS_LLM_MODELS:
        for cond in CROSS_LLM_CONDS:
            for c in range(CFG.n_corpora):
                corpus = load_corpus(m, cond, c)
                if corpus:
                    run_classical(cond, c, "rf", corpus, m)


def stage_transformers():
    print("stage: transformers")
    free_engine()
    gc.collect()
    torch.cuda.empty_cache()
    for cond in ["full", "naive", "only_F5"]:
        for c in range(CFG.n_corpora):
            f = GEN_CACHE / PRIMARY_MODEL.replace("/", "__") / f"{cond}__corpus{c}.json"
            if not f.exists():
                continue
            corpus = json.loads(f.read_text())["headlines"]
            for tm in CFG.transformer_classifiers:
                run_transformer(cond, c, tm, corpus)
                gc.collect()
                torch.cuda.empty_cache()


def stage_baselines():
    print("stage: baselines")
    return run_baselines()


def stage_stats():
    print("stage: stats")
    def agg(clf_name, model_name, condition):
        means = []
        for c in range(CFG.n_corpora):
            f = RESULTS / "classical" / f"{model_name.replace('/','__')}__{condition}__c{c}__{clf_name}.json"
            if f.exists():
                means.append(json.loads(f.read_text())["aggregated"]["f1_fake"]["mean"])
        return np.array(means)

    path = RESULTS / "stats.json"
    if path.exists():
        return json.loads(path.read_text())

    full_means = agg("rf", PRIMARY_MODEL, "full")
    ab = {}
    for fid in FEATURE_IDS:
        m = agg("rf", PRIMARY_MODEL, f"ablate_{fid}")
        if len(m) != len(full_means):
            continue
        d = full_means - m
        _, t_lo, t_hi = t_ci(d)
        ab[fid] = {
            "delta_mean": float(np.mean(d)),
            "delta_std": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "ttest": paired_ttest(d),
            "tost": tost(d, CFG.equivalence_margin),
            "t_ci": [t_lo, t_hi],
            "n_corpora": int(len(d)),
        }
    pvals = [ab[f]["ttest"]["p"] for f in FEATURE_IDS if f in ab]
    padj, rej = fdr(pvals)
    for fid, p, r in zip([f for f in FEATURE_IDS if f in ab], padj, rej):
        ab[fid]["p_fdr"] = float(p)
        ab[fid]["reject_fdr"] = bool(r)
    mm = agg("rf", PRIMARY_MODEL, "only_F5")
    ni = noninf(full_means - mm, CFG.noninf_margin) if len(mm) == len(full_means) else {}
    out = {
        "ablation": ab,
        "noninferiority": ni,
        "f5_note": ("F5 is sufficient (F5-only prompt is non-inferior to full) "
                    "but not proven necessary (removing F5 does not significantly hurt)."),
    }
    path.write_text(json.dumps(out, indent=2, default=str))
    return out


def stage_tokens():
    print("stage: tokens")
    def tok(s):
        return len(s.split())
    f = RESULTS / "token_cost.json"
    if f.exists():
        return json.loads(f.read_text())
    full = build_prompt(FEATURE_IDS, CFG.n_synthetic_per_corpus)
    min5 = build_prompt(["F5"], CFG.n_synthetic_per_corpus)
    naive = build_prompt([], CFG.n_synthetic_per_corpus)
    out = {
        "full_tokens": tok(full),
        "minimal_F5_tokens": tok(min5),
        "naive_tokens": tok(naive),
        "reduction_pct": 100 * (1 - tok(min5) / max(tok(full), 1)),
    }
    f.write_text(json.dumps(out, indent=2))
    return out


def stage_stratified():
    print("stage: stratified")
    f = RESULTS / "stratified.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for cond in ["full", "only_F5"]:
        d = GEN_CACHE / PRIMARY_MODEL.replace("/", "__")
        fp = d / f"{cond}__corpus0.json"
        if not fp.exists():
            continue
        corpus = json.loads(fp.read_text())["headlines"]
        out[cond] = {}
        for src, sp in PER_SOURCE.items():
            out[cond][src] = eval_classical(
                "rf", 0, TEXTS, LABELS, sp["train_idx"], sp["test_idx"], corpus)
    f.write_text(json.dumps(out, indent=2, default=str))
    return out


def _load_all_heads(model_id, cond):
    d = GEN_CACHE / model_id.replace("/", "__")
    out = []
    for c in range(CFG.n_corpora):
        f = d / f"{cond}__corpus{c}.json"
        if f.exists():
            out += json.loads(f.read_text())["headlines"]
    return out


def _rf_f1(model_id, cond):
    vals = []
    for c in range(CFG.n_corpora):
        f = RESULTS / "classical" / f"{model_id.replace('/','__')}__{cond}__c{c}__rf.json"
        if f.exists():
            vals.append(json.loads(f.read_text())["aggregated"]["f1_fake"]["mean"])
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _measure_mean(texts, key):
    if not texts:
        return 0.0
    if key == "excl":
        return float(np.mean([h.count("!") for h in texts]))
    if key == "quest":
        return float(np.mean([h.count("?") for h in texts]))
    if key == "caps":
        return float(np.mean([sum(1 for w in h.split()
                                 if w.isupper() and len(w) > 1) for h in texts]))
    if key == "words":
        return float(np.mean([len(h.split()) for h in texts]))
    raise ValueError(key)


def _fig1_ablation(stats):
    features = [f"F{i}" for i in range(1, 16)]
    deltas = [stats["ablation"][f]["delta_mean"] for f in features]
    t_lo = [stats["ablation"][f]["t_ci"][0] for f in features]
    t_hi = [stats["ablation"][f]["t_ci"][1] for f in features]
    err_low = [d - lo for d, lo in zip(deltas, t_lo)]
    err_high = [hi - d for d, hi in zip(deltas, t_hi)]
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(features, deltas, yerr=[err_low, err_high], capsize=4,
           color=colors, edgecolor="black", linewidth=0.6)
    ax.axhspan(-0.02, 0.02, color="gray", alpha=0.12, zorder=0)
    ax.axhline(0.02, color="black", linestyle="--", linewidth=1.0,
               label=r"Equivalence boundary ($\pm$0.02)")
    ax.axhline(-0.02, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_ylim(-0.023, 0.023)
    ax.set_ylabel(r"$\Delta$F1 (full $-$ ablation)")
    ax.set_title("Impact of removing each stylistic constraint on fake F1 "
                 "(Qwen2.5-7B, 30 seeds \u00d7 3 corpora)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS / "fig1_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()


def _fig2_cross_model():
    families = [
        ("Qwen2.5-7B",     "Qwen/Qwen2.5-7B-Instruct"),
        ("Phi-3.5-mini",   "microsoft/Phi-3.5-mini-instruct"),
        ("DeepSeek-R1-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ]
    family_conds = ["full", "naive", "only_F5"]
    family_labels = {"full": "Full", "naive": "Naive", "only_F5": "F5-only"}
    family_colors = {"full": "#4C72B0", "naive": "#55A868", "only_F5": "#C44E52"}

    suff_conds = ["full", "naive", "only_F5", "only_F6", "only_F12"]
    suff_labels = {"full": "Full", "naive": "Naive", "only_F5": "F5-only",
                   "only_F6": "F6-only", "only_F12": "F12-only"}
    suff_colors = {"full": "#4C72B0", "naive": "#55A868", "only_F5": "#C44E52",
                   "only_F6": "#8172B3", "only_F12": "#CCB974"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    x = np.arange(len(families))
    width = 0.25
    for i, cond in enumerate(family_conds):
        vals, errs = [], []
        for _, m in families:
            mu, sd = _rf_f1(m, cond)
            vals.append(mu if mu is not None else np.nan)
            errs.append(sd if sd is not None else 0)
        pos = x + (i - 1) * width
        bars = ax.bar(pos, vals, width, yerr=errs, capsize=4,
                      label=family_labels[cond], color=family_colors[cond],
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f[0] for f in families])
    ax.set_ylabel("Fake F1")
    ax.set_title("(a) Cross-family prompt conditions", fontsize=12)
    ax.set_ylim(0.575, 0.615)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    x = np.arange(len(suff_conds))
    vals, errs = [], []
    for cond in suff_conds:
        mu, sd = _rf_f1("Qwen/Qwen2.5-7B-Instruct", cond)
        vals.append(mu if mu is not None else np.nan)
        errs.append(sd if sd is not None else 0)
    bars = ax.bar(x, vals, yerr=errs, capsize=4,
                  color=[suff_colors[c] for c in suff_conds],
                  edgecolor="black", linewidth=0.5, width=0.6)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([suff_labels[c] for c in suff_conds])
    ax.set_ylabel("Fake F1")
    ax.set_title("(b) Qwen single-feature sufficiency", fontsize=12)
    ax.set_ylim(0.575, 0.615)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "fig2_cross_model.png", dpi=300, bbox_inches="tight")
    plt.close()


def _fig3_transformers():
    models = [
        ("DistilBERT", "distilbert-base-uncased"),
        ("RoBERTa",    "roberta-base"),
        ("DeBERTa-v3", "microsoft/deberta-v3-base"),
    ]
    conds = [("Full", "full"), ("Naive", "naive"), ("F5-only", "only_F5")]
    colors = {"full": "#4C72B0", "naive": "#55A868", "only_F5": "#C44E52"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, model_id) in zip(axes, models):
        vals, errs = [], []
        for _, cond in conds:
            fs = []
            for c in range(CFG.n_corpora):
                f = RESULTS / "transformers" / f"{cond}__c{c}__{model_id.replace('/','__')}.json"
                if f.exists():
                    fs.append(json.loads(f.read_text())["aggregated"]["f1_fake"]["mean"])
            vals.append(float(np.mean(fs)) if fs else np.nan)
            errs.append(float(np.std(fs)) if fs else 0)
        x = np.arange(len(conds))
        bars = ax.bar(x, vals, yerr=errs, capsize=4,
                      color=[colors[c[1]] for c in conds],
                      edgecolor="black", linewidth=0.5, width=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in conds])
        ax.set_ylabel("Fake F1")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0.685, 0.72)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "fig3_transformers.png", dpi=300, bbox_inches="tight")
    plt.close()


def _fig4_ratio():
    qwen_id = "Qwen/Qwen2.5-7B-Instruct"
    conds = ["full", "naive", "only_F5", "only_F6", "only_F12"]
    labels = {"full": "Full", "naive": "Naive", "only_F5": "F5-only",
              "only_F6": "F6-only", "only_F12": "F12-only"}
    colors = {"full": "#4C72B0", "naive": "#55A868", "only_F5": "#C44E52",
              "only_F6": "#8172B3", "only_F12": "#CCB974"}
    keys = ["excl", "quest", "caps", "words"]
    key_labels = {"excl": "Exclamation marks", "quest": "Question marks",
                  "caps": "ALL CAPS words", "words": "Word count"}

    df_real = pd.read_csv(DATASET_CACHE / "headlines.csv")
    real_fake = df_real[df_real["label"] == 1]["text"].tolist()
    target = {k: _measure_mean(real_fake, k) for k in keys}

    synth_means = {}
    for cond in conds:
        texts = _load_all_heads(qwen_id, cond)
        synth_means[cond] = {k: _measure_mean(texts, k) for k in keys}

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(keys))
    width = 0.16
    for i, cond in enumerate(conds):
        ratios = []
        for k in keys:
            tgt = target[k] if target[k] > 1e-6 else 1.0
            ratios.append(synth_means[cond][k] / tgt)
        pos = x + (i - 2) * width
        bars = ax.bar(pos, ratios, width, label=labels[cond],
                      color=colors[cond], edgecolor="black", linewidth=0.5)
        for b, r in zip(bars, ratios):
            ax.text(b.get_x() + b.get_width()/2, r, f"{r:.2f}x",
                    ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="Perfect match (1.0x)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([key_labels[k] for k in keys], rotation=15, ha="right")
    ax.set_ylabel("Synthetic mean / Real fake mean (log scale)")
    ax.set_title("Ratio of synthetic to real fake-news values "
                 "(Qwen2.5-7B, 3 corpora per condition)")
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS / "target_vs_synth_ratio.png", dpi=300, bbox_inches="tight")
    plt.close()


def _fig5_heatmap():
    rows = [
        ("Qwen2.5-7B",     "Qwen/Qwen2.5-7B-Instruct",              "full"),
        ("Qwen2.5-7B",     "Qwen/Qwen2.5-7B-Instruct",              "naive"),
        ("Qwen2.5-7B",     "Qwen/Qwen2.5-7B-Instruct",              "only_F5"),
        ("Qwen2.5-7B",     "Qwen/Qwen2.5-7B-Instruct",              "only_F6"),
        ("Qwen2.5-7B",     "Qwen/Qwen2.5-7B-Instruct",              "only_F12"),
        ("Phi-3.5-mini",   "microsoft/Phi-3.5-mini-instruct",        "full"),
        ("Phi-3.5-mini",   "microsoft/Phi-3.5-mini-instruct",        "naive"),
        ("Phi-3.5-mini",   "microsoft/Phi-3.5-mini-instruct",        "only_F5"),
        ("DeepSeek-R1-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "full"),
        ("DeepSeek-R1-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "naive"),
        ("DeepSeek-R1-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "only_F5"),
    ]
    labels = {"full": "Full", "naive": "Naive", "only_F5": "F5-only",
              "only_F6": "F6-only", "only_F12": "F12-only"}
    keys = ["excl", "quest", "caps", "words"]
    key_labels = {"excl": "Exclamation\nmarks", "quest": "Question\nmarks",
                  "caps": "ALL CAPS\nwords", "words": "Words per\nheadline"}

    df_real = pd.read_csv(DATASET_CACHE / "headlines.csv")
    real_fake = df_real[df_real["label"] == 1]["text"].tolist()
    target = {k: _measure_mean(real_fake, k) for k in keys}

    matrix_rows = []
    row_labels = []
    for model_label, model_id, cond in rows:
        texts = _load_all_heads(model_id, cond)
        r = []
        for k in keys:
            tgt = target[k] if target[k] > 1e-6 else 1.0
            r.append(_measure_mean(texts, k) / tgt if texts else np.nan)
        matrix_rows.append(r)
        row_labels.append(f"{model_label}\n({labels[cond]})")

    matrix = np.array(matrix_rows, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_matrix = np.log2(matrix)
    finite = log2_matrix[~np.isnan(log2_matrix)]
    vmax_abs = float(np.max(np.abs(finite))) if len(finite) else 1.0
    vmax_abs = max(vmax_abs, 1.0)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)

    fig, ax = plt.subplots(figsize=(8.5, 10))
    sns.heatmap(
        log2_matrix, cmap=cmap, norm=norm,
        xticklabels=[key_labels[k] for k in keys],
        yticklabels=row_labels,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "log2(synthetic / real fake)", "shrink": 0.6},
        ax=ax,
    )
    for i in range(log2_matrix.shape[0]):
        for j in range(log2_matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                txt, color = "n/a", "black"
            else:
                txt = f"{v:.2f}x"
                color = "white" if abs(log2_matrix[i, j]) > 0.5 * vmax_abs else "black"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")
    for y in [5, 8]:
        ax.axhline(y, color="black", lw=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS / "cross_family_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def stage_figures(stats, tokens):
    print("stage: figures")
    _fig1_ablation(stats)
    _fig2_cross_model()
    _fig3_transformers()
    _fig4_ratio()
    _fig5_heatmap()


def stage_report(stats, tokens, strat, baselines):
    print("stage: report")
    def load_all(m, cond):
        return _load_all_heads(m, cond)

    L = []
    L.append("# Feature Ablation Experiment")
    L.append("")
    L.append(f"Generated: {datetime.now().isoformat()}")
    L.append("")
    L.append(f"Primary generator: {PRIMARY_MODEL}")
    L.append(f"Cross-LLM models: {CROSS_LLM_MODELS}")
    L.append(f"Corpora per condition: {CFG.n_corpora}")
    L.append(f"Classifier seeds: {CFG.classifier_seeds}")
    L.append("")
    L.append("## Baselines")
    for name, agg in baselines.items():
        L.append(f"- {name}: fake F1 = {agg['f1_fake']['mean']:.4f} +/- {agg['f1_fake']['std']:.4f}")
    L.append("")
    L.append("## Ablation")
    L.append("")
    L.append("| Feature | F1 diff | p (t) | p (FDR) | TOST p | Equivalent? | 95% CI |")
    L.append("|---|---|---|---|---|---|---|")
    for fid in FEATURE_IDS:
        a = stats["ablation"].get(fid)
        if not a:
            continue
        ci = f"[{a['t_ci'][0]:.4f}, {a['t_ci'][1]:.4f}]"
        L.append(f"| {fid} | {a['delta_mean']:.4f} | {a['ttest']['p']:.4f} "
                 f"| {a.get('p_fdr', float('nan')):.4f} | {a['tost']['p_equiv']:.4f} "
                 f"| {'Yes' if a['tost']['equiv_at_alpha'] else 'No'} | {ci} |")
    L.append("")
    ni = stats.get("noninferiority", {})
    L.append("## Non-inferiority (full vs F5-only)")
    L.append(f"- F1 difference = {ni.get('mean', float('nan')):.4f}, "
             f"p = {ni.get('p', float('nan')):.4f}, margin = {CFG.noninf_margin}")
    L.append(f"- Non-inferior: {ni.get('noninferior_at_alpha', False)}")
    L.append("")
    L.append(f"**{stats.get('f5_note', '')}**")
    L.append("")
    L.append("## Compliance table")
    full_corpus = load_all(PRIMARY_MODEL, "full")
    full_corpus = full_corpus[:CFG.n_synthetic_per_corpus] if full_corpus else []
    if full_corpus:
        comp = feature_compliance_summary(full_corpus)
        L.append("| Feature | Compliance | Band |")
        L.append("|---|---|---|")
        for fid in FEATURE_IDS:
            lo, hi = BANDS[fid]
            L.append(f"| {fid} | {comp[fid]*100:5.1f}% | [{lo}, {hi}] |")
        avg = float(np.mean(list(comp.values()))) * 100
        L.append(f"| Average | {avg:.1f}% | - |")
    L.append("")
    L.append("## Target vs synthetic distribution")
    try:
        df_real = pd.read_csv(DATASET_CACHE / "headlines.csv")
        real_fake = df_real[df_real["label"] == 1]["text"].tolist()
        target = {k: _measure_mean(real_fake, k) for k in ["excl", "quest", "caps", "words"]}
        L.append("| Feature | Real Fake | Full | Naive | F5-only | F6-only | F12-only |")
        L.append("|---|---|---|---|---|---|---|")
        for k in ["excl", "quest", "caps", "words"]:
            row = [f"{_measure_mean(load_all(PRIMARY_MODEL, c), k):.3f}"
                   for c in ["full", "naive", "only_F5", "only_F6", "only_F12"]]
            L.append(f"| {k} | {target[k]:.3f} | " + " | ".join(row) + " |")
        L.append("")
        L.append("| Feature | Full | Naive | F5-only | F6-only | F12-only |")
        L.append("|---|---|---|---|---|---|")
        for k in ["excl", "quest", "caps", "words"]:
            tgt = target[k] if target[k] > 1e-6 else 1.0
            row = [f"{_measure_mean(load_all(PRIMARY_MODEL, c), k)/tgt:.2f}x"
                   for c in ["full", "naive", "only_F5", "only_F6", "only_F12"]]
            L.append(f"| {k} | " + " | ".join(row) + " |")
    except Exception as e:
        L.append(f"(error: {e})")
    L.append("")
    L.append("## Cross-family compliance ratios")
    try:
        df_real = pd.read_csv(DATASET_CACHE / "headlines.csv")
        real_fake = df_real[df_real["label"] == 1]["text"].tolist()
        target = {k: _measure_mean(real_fake, k) for k in ["excl", "quest", "caps", "words"]}
        L.append("| Model | Feature | Full | Naive | F5-only |")
        L.append("|---|---|---|---|---|")
        for m in CROSS_LLM_MODELS + [PRIMARY_MODEL]:
            short = m.split("/")[-1]
            for k in ["excl", "quest", "caps", "words"]:
                tgt = target[k] if target[k] > 1e-6 else 1.0
                row = []
                for cond in ["full", "naive", "only_F5"]:
                    texts = load_all(m, cond)
                    row.append(f"{_measure_mean(texts, k)/tgt:.2f}x" if texts else "n/a")
                L.append(f"| {short} | {k} | {row[0]} | {row[1]} | {row[2]} |")
    except Exception as e:
        L.append(f"(error: {e})")
    L.append("")
    L.append("## Transformer results")
    tf_dir = RESULTS / "transformers"
    for cond in ["full", "naive", "only_F5"]:
        for c in range(CFG.n_corpora):
            for tm in CFG.transformer_classifiers:
                safe = tm.replace("/", "__")
                fp = tf_dir / f"{cond}__c{c}__{safe}.json"
                if fp.exists():
                    r = json.loads(fp.read_text())["aggregated"]
                    L.append(f"- {cond} c{c} {tm}: fake F1 = {r['f1_fake']['mean']:.4f} "
                             f"+/- {r['f1_fake']['std']:.4f}")
    L.append("")
    L.append("## LLM classifier")
    for m in [CFG.llm_classifier_model, PRIMARY_MODEL]:
        fp = RESULTS / f"llm_classifier__{m.replace('/', '__')}.json"
        if fp.exists():
            L.append(json.dumps(json.loads(fp.read_text()), indent=2))
            break
    L.append("")
    L.append("## Token table")
    L.append(json.dumps(tokens, indent=2))
    L.append("")
    L.append("## Stratified analysis")
    L.append(json.dumps(strat, indent=2, default=str)[:3000])
    L.append("")
    L.append("## Cross-LLM generation")
    for m in CROSS_LLM_MODELS:
        L.append(f"- {m}: {CROSS_LLM_CONDS}")
    out = "\n".join(L)
    (REPORTS / "final_report.md").write_text(out)
    return out


def print_results():
    print()
    print("=" * 78)
    print("COMPLETE RESULTS")
    print("=" * 78)

    print("")
    print("--- Baselines ---")
    if (RESULTS / "baselines.json").exists():
        b = json.loads((RESULTS / "baselines.json").read_text())
        for name, agg in b.items():
            print(f"  {name:14s} fake F1 = {agg['f1_fake']['mean']:.4f} "
                  f"+/- {agg['f1_fake']['std']:.4f}")

    print("")
    print("--- Cross-family prompt conditions (RF fake F1) ---")
    families = {
        "Qwen2.5-7B":     "Qwen/Qwen2.5-7B-Instruct",
        "Phi-3.5-mini":   "microsoft/Phi-3.5-mini-instruct",
        "DeepSeek-R1-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    }
    print(f"  {'Model':20s} {'Full':>10s} {'Naive':>10s} {'F5-only':>10s}")
    for label, m in families.items():
        row = []
        for cond in ["full", "naive", "only_F5"]:
            vals = []
            for c in range(CFG.n_corpora):
                f = RESULTS / "classical" / f"{m.replace('/','__')}__{cond}__c{c}__rf.json"
                if f.exists():
                    vals.append(json.loads(f.read_text())["aggregated"]["f1_fake"]["mean"])
            row.append(f"{np.mean(vals):.4f}" if vals else "-")
        print(f"  {label:20s} {row[0]:>10s} {row[1]:>10s} {row[2]:>10s}")

    print("")
    print("--- Ablation (RF, primary model) ---")
    if (RESULTS / "stats.json").exists():
        s = json.loads((RESULTS / "stats.json").read_text())
        print(f"  {'Feature':8s} {'dF1':>10s} {'p(t)':>10s} {'q(FDR)':>10s} "
              f"{'TOST p':>10s} {'Eqv':>5s}  {'t-CI':>24s}")
        for fid in FEATURE_IDS:
            a = s["ablation"].get(fid)
            if not a:
                continue
            ci = f"[{a['t_ci'][0]:+.4f},{a['t_ci'][1]:+.4f}]"
            print(f"  {fid:8s} {a['delta_mean']:>+10.4f} {a['ttest']['p']:>10.4f} "
                  f"{a.get('p_fdr', float('nan')):>10.4f} {a['tost']['p_equiv']:>10.4f} "
                  f"{'Yes' if a['tost']['equiv_at_alpha'] else 'No':>5s}  {ci:>24s}")
        ni = s.get("noninferiority", {})
        print("")
        print(f"  Non-inferiority full vs F5-only: "
              f"dF1 = {ni.get('mean', float('nan')):+.4f}, "
              f"p = {ni.get('p', float('nan')):.4f}, margin = {CFG.noninf_margin}")

    print("")
    print("--- Sufficiency and interaction conditions ---")
    full_vals = []
    for c in range(CFG.n_corpora):
        f = RESULTS / "classical" / f"{PRIMARY_MODEL.replace('/','__')}__full__c{c}__rf.json"
        if f.exists():
            full_vals.append(json.loads(f.read_text())["aggregated"]["f1_fake"]["mean"])
    full_f1 = float(np.mean(full_vals)) if full_vals else float("nan")
    conds = ["full", "naive", "only_F5", "only_F6", "only_F12",
             "only_F5_F6", "only_F5_F4", "only_F6_F4",
             "only_F5_F12", "only_F5_F6_F4", "only_F5_F6_F12"]
    for cond in conds:
        vals = []
        for c in range(CFG.n_corpora):
            f = RESULTS / "classical" / f"{PRIMARY_MODEL.replace('/','__')}__{cond}__c{c}__rf.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["aggregated"]["f1_fake"]["mean"])
        if vals:
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            delta = full_f1 - mu
            print(f"  {cond:20s} {mu:.4f} +/- {sd:.4f}  dF1 = {delta:+.4f}")

    print("")
    print("--- Compliance (full prompt, first corpus) ---")
    full_path = GEN_CACHE / PRIMARY_MODEL.replace("/", "__") / "full__corpus0.json"
    if full_path.exists():
        corpus = json.loads(full_path.read_text())["headlines"]
        comp = feature_compliance_summary(corpus)
        for fid in FEATURE_IDS:
            print(f"  {fid:5s} {comp[fid]*100:6.1f}%")
        avg = float(np.mean(list(comp.values()))) * 100
        print(f"  AVG   {avg:6.1f}%")

    print("")
    print("--- Target vs synthetic (primary model) ---")
    df_real = pd.read_csv(DATASET_CACHE / "headlines.csv")
    real_fake = df_real[df_real["label"] == 1]["text"].tolist()
    target = {k: _measure_mean(real_fake, k) for k in ["excl", "quest", "caps", "words"]}
    print(f"  {'Feature':10s} {'RealFake':>10s} {'Full':>10s} {'Naive':>10s} {'F5':>10s} {'F6':>10s} {'F12':>10s}")
    for k in ["excl", "quest", "caps", "words"]:
        row = [_measure_mean(_load_all_heads(PRIMARY_MODEL, c), k)
               for c in ["full", "naive", "only_F5", "only_F6", "only_F12"]]
        print(f"  {k:10s} {target[k]:>10.3f} " + " ".join(f"{v:>10.3f}" for v in row))
    print(f"  {'Feature':10s} {'Full':>10s} {'Naive':>10s} {'F5':>10s} {'F6':>10s} {'F12':>10s}   (ratios)")
    for k in ["excl", "quest", "caps", "words"]:
        tgt = target[k] if target[k] > 1e-6 else 1.0
        row = [_measure_mean(_load_all_heads(PRIMARY_MODEL, c), k) / tgt
               for c in ["full", "naive", "only_F5", "only_F6", "only_F12"]]
        print(f"  {k:10s} " + " ".join(f"{v:>9.2f}x" for v in row))

    print("")
    print("--- Cross-family compliance ratios ---")
    for label, m in families.items():
        for k in ["excl", "quest", "caps", "words"]:
            tgt = target[k] if target[k] > 1e-6 else 1.0
            row = [_measure_mean(_load_all_heads(m, c), k) / tgt
                   for c in ["full", "naive", "only_F5"]]
            print(f"  {label:18s} {k:6s} " + " ".join(f"{v:>8.2f}x" for v in row))
        print("")

    print("--- Transformer results (primary model corpora) ---")
    for tm in CFG.transformer_classifiers:
        row = []
        for cond in ["full", "naive", "only_F5"]:
            vals = []
            for c in range(CFG.n_corpora):
                f = RESULTS / "transformers" / f"{cond}__c{c}__{tm.replace('/','__')}.json"
                if f.exists():
                    vals.append(json.loads(f.read_text())["aggregated"]["f1_fake"]["mean"])
            row.append(f"{np.mean(vals):.4f}" if vals else "-")
        print(f"  {tm:35s} {row[0]:>8s} {row[1]:>8s} {row[2]:>8s}")

    print("")
    print("--- Zero-shot LLM classifier ---")
    for m in [CFG.llm_classifier_model, PRIMARY_MODEL]:
        f = RESULTS / f"llm_classifier__{m.replace('/','__')}.json"
        if f.exists():
            print(json.dumps(json.loads(f.read_text()), indent=2))
            break

    print("")
    print("--- Tokens ---")
    if (RESULTS / "token_cost.json").exists():
        print(json.dumps(json.loads((RESULTS / "token_cost.json").read_text()), indent=2))

    print("")
    print("--- Stratified ---")
    if (RESULTS / "stratified.json").exists():
        s = json.loads((RESULTS / "stratified.json").read_text())
        for cond, sources in s.items():
            for src, r in sources.items():
                print(f"  {cond:10s} {src:12s} F1 = {r['f1_fake']:.4f}")

    print("")
    print("=" * 78)
    print("END OF RESULTS")
    print("=" * 78)


def stage_package():
    print("stage: package")
    staging = ROOT / "paper_artifacts"
    out_zip = ROOT / "paper_artifacts.zip"
    if staging.exists():
        shutil.rmtree(staging)
    if out_zip.exists():
        out_zip.unlink()
    staging.mkdir(parents=True)
    (staging / "reports").mkdir()
    shutil.copy(CACHE / "reports" / "final_report.md",
                staging / "reports" / "final_report.md")
    (staging / "results").mkdir()
    for f in (CACHE / "results").glob("*.json"):
        shutil.copy(f, staging / "results" / f.name)
    shutil.copytree(CACHE / "results" / "classical", staging / "results" / "classical")
    shutil.copytree(CACHE / "results" / "transformers", staging / "results" / "transformers")
    shutil.copytree(CACHE / "generated", staging / "generated")
    shutil.copytree(CACHE / "plots", staging / "plots")
    shutil.copytree(CACHE / "dataset", staging / "dataset")
    shutil.copy(CACHE / "results" / "metadata.json", staging / "metadata.json")

    readme = staging / "README.md"
    readme.write_text("""# Synthetic Fake News Corpus Ablation

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
""")
    shutil.make_archive(str(out_zip.with_suffix("")), "zip",
                        staging.parent, staging.name)
    print(f"archive: {out_zip}")
    print(f"size: {out_zip.stat().st_size / 1e6:.1f} MB")


def main():
    stage_generation()
    stage_llm_classifier()
    stage_classical()
    stage_transformers()
    baselines = stage_baselines()
    stats = stage_stats()
    tokens = stage_tokens()
    strat = stage_stratified()
    stage_figures(stats, tokens)
    stage_report(stats, tokens, strat, baselines)
    print_results()
    stage_package()
    print("done")


if __name__ == "__main__":
    main()