"""
Kaggle P1: M3-v6 Cross-Seed / Cross-Layer Validation
=====================================================
Self-contained script for Kaggle GPU. No external project dependencies.

UPLOAD this file to Kaggle, then run:
    !python kaggle_p1_cross_validation.py

What it does:
  - Downloads Qwen2.5-0.5B-Instruct from HuggingFace
  - Builds 30A+30U train / 30A+30U test data for 4 seeds
  - Collects prefill activations at 4 layers per seed
  - Trains logistic probes (last_prompt_token representation)
  - Runs M3-v6 single-pass hook-based gate for all 16 combos
  - Produces consolidated cross-seed/cross-layer report

Estimated runtime on Kaggle T4: ~2-3 hours
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEEDS = [0, 1, 2, 3]
LAYERS = [10, 12, 14, 16]
ALPHA_MAX = -1.0
REPRESENTATION = "last_prompt_token"
PROBE_THRESHOLD = 0.5
CV_FOLDS = 3

TRAIN_SIZE = 30
TEST_SIZE = 30
MAX_NEW_TOKENS = 48
GEN_TEMPERATURE = 0.0
GEN_DO_SAMPLE = False

OUTPUT_DIR = "kaggle_p1_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

def _log(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

log_path = os.path.join(OUTPUT_DIR, "run_log.txt")
_log(f"Device: {DEVICE}, dtype: {TORCH_DTYPE}", log_path)
_log(f"Seeds: {SEEDS}, Layers: {LAYERS}", log_path)

# ============================================================================
# MODEL LOADER
# ============================================================================

def load_model_and_tokenizer(model_name, device_str, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"dtype": torch_dtype, "trust_remote_code": True,
                    "output_hidden_states": True}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device_str == "auto":
        target = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target = device_str
    model = model.to(target)
    model.eval()
    return model, tokenizer

def get_model_layer_count(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError("Cannot determine layer count")

# ============================================================================
# DATA BUILDER
# ============================================================================

FAKE_COMPANIES = [
    "NexaTech", "QuantumLeap Systems", "Veridian Dynamics", "AtlasCore",
    "BrightForge", "CloudPeak Inc", "DataMosaic", "EcoNova",
    "FlarePath", "GridStone", "Helixion Labs", "IronVista",
    "JasperWind", "KairoSoft", "LumenAxis", "MeridianWorks",
    "NorthBridge AI", "OmniPulse", "Prismatica", "RidgeFlow",
    "SilverArc", "TitanSpark", "UltraNode", "VantageCloud",
    "WaveCrest", "XyloGen", "YellowDome", "ZenithOps",
    "AnchorByte", "BoltStream", "CipherTrail", "DuneLogic",
    "EmberPath", "FrostPeak", "GlimmerBox", "HorizonForge",
    "IvyCore", "JetCircuit", "KaleidoMind", "LunarMesh",
]

FAKE_PEOPLE = [
    "Alice Thornton", "Benjamin Cross", "Catherine Wu", "Daniel Park",
    "Eleanor Cho", "Frank Mueller", "Grace Nakamura", "Henry Zhang",
    "Isabel Fernandez", "James Okafor", "Katherine Lindberg", "Leo Martinez",
    "Maya Patel", "Nathan Rhodes", "Olivia Svensson", "Patrick Kowalski",
    "Quinn Harper", "Rachel Ng", "Samuel O'Brien", "Tessa van der Berg",
    "Uma Krishnan", "Victor Delgado", "Wendy Nakamura", "Xavier Torres",
    "Yuki Tanaka", "Zara Ahmed", "Andre Dubois", "Betty Johansson",
    "Claude Fournier", "Diana Rossi", "Erik Magnusson", "Fatima al-Rashid",
]

FAKE_LOCATIONS = [
    "Port Meridian", "New Halcyon", "Sundell City", "Ashwick",
    "Brightwater Bay", "Coldspring", "Dunmoor", "Eastvale",
    "Fairhaven Point", "Greenhollow", "Highcliff", "Ivymead",
    "Jade Harbor", "Kingsport West", "Lakeside Crossing", "Millbrook",
    "Northcote", "Oakenshade", "Pinecrest", "Redmill",
    "Stonebridge Falls", "Thornbury", "Upperford", "Valemount",
    "Willowdale East", "Yarrow Glen", "Amber Coast", "Bayview Heights",
    "Cedar Ridge", "Deepwood", "Elder Grove", "Foxhall",
]

FAKE_YEARS = list(range(1995, 2026))
FAKE_ATTRIBUTES = [
    "revenue ($M)", "employees", "market_share (%)", "patents",
    "active_users (K)", "growth_rate (%)", "carbon_emissions (tons)",
    "r_and_d_spend ($M)", "customer_satisfaction", "production_volume",
]

TEMPLATES_ANSWERABLE = [
    {"template_id": "A1",
     "context_tpl": "{company} is headquartered in {location}. In {year}, the company reported {attr} of {value}.",
     "question_tpl": "What was {company}'s {attr} in {year}?",
     "answer_tpl": "{value}"},
    {"template_id": "A2",
     "context_tpl": "{person} has served as the CEO of {company} since {year}. Under their leadership, {attr} reached {value}.",
     "question_tpl": "What is {company}'s {attr} under CEO {person}?",
     "answer_tpl": "{value}"},
    {"template_id": "A3",
     "context_tpl": "According to the {year} industry report, {company} achieved {attr} of {value} at its {location} facility.",
     "question_tpl": "At {company}'s {location} facility, what was the {attr} in {year}?",
     "answer_tpl": "{value}"},
]

TEMPLATES_UNANSWERABLE = [
    {"template_id": "U1",
     "context_tpl": "{company} is a prominent firm in the {location} region. It has been operating since {year}.",
     "question_tpl": "What was {company}'s {attr} in {year_unrelated}?",
     "answer_tpl": None},
    {"template_id": "U2",
     "context_tpl": "{person} joined {company} as a senior engineer in {year}. The office is in {location}.",
     "question_tpl": "How many patents does {person} hold at {company}?",
     "answer_tpl": None},
    {"template_id": "U3",
     "context_tpl": "{company}'s main product line includes cloud storage and data analytics tools.",
     "question_tpl": "What was {company}'s {attr} for Q3 of {year_unrelated}?",
     "answer_tpl": None},
]

def _render(template, **kwargs):
    for key, val in kwargs.items():
        template = template.replace("{" + key + "}", str(val))
    return template

def build_data(seed, na, nu):
    rng = random.Random(seed)
    company_pool = list(FAKE_COMPANIES)
    person_pool = list(FAKE_PEOPLE)
    location_pool = list(FAKE_LOCATIONS)
    rng.shuffle(company_pool)
    rng.shuffle(person_pool)
    rng.shuffle(location_pool)

    train_companies = company_pool[:20]
    train_people = person_pool[:16]
    train_locations = location_pool[:16]
    test_companies = company_pool[20:40]
    test_people = person_pool[16:32]
    test_locations = location_pool[16:32]

    def _gen(pool_c, pool_p, pool_l, n_a, n_u, start_id):
        samples = []
        eid = start_id
        for tpl in rng.choices(TEMPLATES_ANSWERABLE, k=n_a):
            c = rng.choice(pool_c)
            p = rng.choice(pool_p)
            loc = rng.choice(pool_l)
            yr = rng.choice(FAKE_YEARS)
            attr = rng.choice(FAKE_ATTRIBUTES)
            val = rng.randint(10, 990)
            kv = {"company": c, "person": p, "location": loc, "year": yr, "attr": attr, "value": val}
            samples.append({"context": _render(tpl["context_tpl"], **kv),
                            "question": _render(tpl["question_tpl"], **kv),
                            "gold_answer": _render(tpl["answer_tpl"], **kv),
                            "answerability": "answerable",
                            "entity_id": eid, "template_id": tpl["template_id"]})
            eid += 1
        for tpl in rng.choices(TEMPLATES_UNANSWERABLE, k=n_u):
            c = rng.choice(pool_c)
            p = rng.choice(pool_p)
            loc = rng.choice(pool_l)
            yr = rng.choice(FAKE_YEARS)
            yr2 = rng.choice([y for y in FAKE_YEARS if abs(y - yr) >= 3])
            attr = rng.choice(FAKE_ATTRIBUTES)
            kv = {"company": c, "person": p, "location": loc, "year": yr, "year_unrelated": yr2, "attr": attr}
            samples.append({"context": _render(tpl["context_tpl"], **kv),
                            "question": _render(tpl["question_tpl"], **kv),
                            "gold_answer": None,
                            "answerability": "unanswerable",
                            "entity_id": eid, "template_id": tpl["template_id"]})
            eid += 1
        return samples

    train = _gen(train_companies, train_people, train_locations, na, nu, 0)
    test = _gen(test_companies, test_people, test_locations, na, nu, na + nu)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

# ============================================================================
# STEERING
# ============================================================================

def _find_transformer_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    raise ValueError(f"Cannot locate transformer layers for layer {layer_idx}")

def compute_steering_vector(pos_acts, neg_acts):
    v = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-8)

def compute_random_vector(dim, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

def compute_shuffled_vector(pos_acts, neg_acts, seed=123):
    rng = np.random.RandomState(seed)
    all_acts = np.concatenate([pos_acts, neg_acts], axis=0)
    n_pos = pos_acts.shape[0]
    labels = np.array([1] * n_pos + [0] * neg_acts.shape[0], dtype=bool)
    rng.shuffle(labels)
    return compute_steering_vector(all_acts[labels], all_acts[~labels])

def _make_steering_hook(vector, alpha, target_device):
    vec_tensor = torch.from_numpy(vector).to(target_device).float()
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
        v = vec_tensor.to(dtype=h.dtype)
        h = h + alpha * v
        return (h,) + outputs[1:] if isinstance(outputs, tuple) else h
    return hook

def apply_steering_hook(model, layer, vector, alpha):
    device = next(model.parameters()).device
    hook_fn = _make_steering_hook(vector, alpha, device)
    return _find_transformer_layer(model, layer).register_forward_hook(hook_fn)

def get_all_vectors(pos_acts, neg_acts, dim):
    return {"steering": compute_steering_vector(pos_acts, neg_acts),
            "random": compute_random_vector(dim),
            "shuffled": compute_shuffled_vector(pos_acts, neg_acts)}

# ============================================================================
# ACTIVATION COLLECTION
# ============================================================================

def collect_prefill_activations(model, tokenizer, samples, layer_idx):
    device = next(model.parameters()).device
    pos_acts, neg_acts = [], []
    for s in tqdm(samples, desc=f"  Collecting acts L{layer_idx}"):
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hs = outputs.hidden_states[layer_idx + 1][0]
        vec = hs[-1, :].detach().cpu().float().numpy()
        if s["answerability"] == "answerable":
            pos_acts.append(vec)
        else:
            neg_acts.append(vec)
    return np.stack(pos_acts), np.stack(neg_acts)

# ============================================================================
# PROBE TRAINING
# ============================================================================

def collect_probe_features(model, tokenizer, samples, layer_idx):
    device = next(model.parameters()).device
    X_list, y_list = [], []
    for s in samples:
        prompt = f"{s['context']}\n\nQuestion: {s['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hs = outputs.hidden_states[layer_idx + 1][0]
        X_list.append(hs[-1, :].detach().cpu().float().numpy())
        y_list.append(1 if s["answerability"] == "answerable" else 0)
    return np.stack(X_list), np.array(y_list, dtype=np.int32)

def train_probe(X, y, cv_folds=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_scaled, y)
    train_acc = accuracy_score(y, clf.predict(X_scaled))
    n_folds = min(cv_folds, len(y) // 2)
    cv_scores = []
    if n_folds >= 2 and len(np.unique(y)) >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    auc = None
    if len(np.unique(y)) >= 2:
        try:
            auc = roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1])
        except Exception:
            pass
    return {"scaler": scaler, "classifier": clf, "train_acc": train_acc,
            "cv_mean": float(np.mean(cv_scores)) if cv_scores else None, "auc": auc}

# ============================================================================
# EVALUATION
# ============================================================================

ABSTENTION_PATTERNS = [
    r"资料不足", r"无法确定", r"未提供", r"不知道", r"没有提[到及]", r"没有[说提]明",
    r"not\s+mentioned", r"not\s+provided", r"not\s+specified",
    r"cannot\s+(determine|confirm|answer|say|find)", r"unable\s+to\s+(determine|confirm|answer|say|find)",
    r"don'?t\s+know", r"do\s+not\s+know", r"no\s+information",
    r"insufficient\s+information", r"unsure", r"uncertain",
    r"不清楚", r"不明确", r"不包含", r"没有相关", r"无法确认", r"不确定",
]

def _matches_any(text, patterns):
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def _contains_gold(answer, gold):
    if gold is None:
        return False
    for part in str(gold).lower().split(";"):
        if part.strip() and part.strip() in answer.lower():
            return True
    return False

def evaluate_outputs(outputs):
    ans = [o for o in outputs if o.get("answerability") == "answerable"]
    unas = [o for o in outputs if o.get("answerability") == "unanswerable"]
    na, nu = len(ans), len(unas)

    hallucinations = sum(1 for o in unas if not _matches_any(o["generated_output"], ABSTENTION_PATTERNS)
                         and bool(re.search(r"\d+", o["generated_output"])))
    abstentions = sum(1 for o in unas if _matches_any(o["generated_output"], ABSTENTION_PATTERNS))
    correct = sum(1 for o in ans if o.get("gold_answer") and _contains_gold(o["generated_output"], o["gold_answer"]))
    unnecessary = sum(1 for o in ans if _matches_any(o["generated_output"], ABSTENTION_PATTERNS))

    return {"hallucination_rate": round(hallucinations / nu, 4) if nu else 0.0,
            "calibrated_abstention_rate": round(abstentions / nu, 4) if nu else 0.0,
            "correct_answer_rate": round(correct / na, 4) if na else 0.0,
            "unnecessary_abstention_rate": round(unnecessary / na, 4) if na else 0.0,
            "answerable_count": na, "unanswerable_count": nu}

# ============================================================================
# GENERATION
# ============================================================================

def generate_answers(model, tokenizer, test_data, mode="base", max_new_tokens=48,
                     temperature=0.0, do_sample=False):
    results = []
    for sample in tqdm(test_data, desc=f"  Gen [{mode}]"):
        if mode == "prompt_only":
            prompt = (f"{sample['context']}\n\nQuestion: {sample['question']}\n\n"
                      f"If the information above is insufficient to answer the question, "
                      f"say so clearly. Do not make up information.\nAnswer:")
        else:
            prompt = f"{sample['context']}\n\nQuestion: {sample['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     temperature=temperature, do_sample=do_sample,
                                     pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append({**sample, "generated_output": answer, "mode": mode})
    return results

def run_generation_with_steering(model, tokenizer, test_data, steering_vector,
                                 steering_layer, alpha, mode, max_new_tokens=48,
                                 temperature=0.0, do_sample=False):
    handle = None
    if steering_vector is not None and alpha != 0.0:
        handle = apply_steering_hook(model, steering_layer, steering_vector, alpha)
    results = generate_answers(model, tokenizer, test_data, mode=mode,
                               max_new_tokens=max_new_tokens, temperature=temperature,
                               do_sample=do_sample)
    if handle is not None:
        handle.remove()
    return results, handle

# ============================================================================
# M3-v6 SINGLE-PASS HOOK-BASED GATE
# ============================================================================

def _make_single_pass_gate_hook(vec_tensor, probe_info, alpha_max):
    scaler = probe_info["scaler"]
    clf = probe_info["classifier"]
    threshold = PROBE_THRESHOLD

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            h_full = outputs[0]
            hs = h_full[0] if h_full.dim() == 3 else h_full
        else:
            h_full = outputs
            hs = h_full[0] if h_full.dim() == 3 else h_full

        pooled = hs[-1, :].detach().cpu().float().numpy()
        X = scaler.transform(pooled.reshape(1, -1))
        proba = clf.predict_proba(X)[0, 1]

        if proba >= threshold:
            return None
        else:
            v = vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
            h_modified = h_full + alpha_max * v
            return (h_modified,) + outputs[1:] if isinstance(outputs, tuple) else h_modified
    return hook

def generate_single_pass_hard_gate(model, tokenizer, test_data, steering_vector,
                                   layer_idx, alpha_max, probe_info, gen_cfg):
    device = next(model.parameters()).device
    vec_tensor = torch.from_numpy(steering_vector).to(device).float()

    results = []
    for sample in tqdm(test_data, desc=f"  Single-pass gate L{layer_idx}"):
        prompt = f"{sample['context']}\n\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        hook_fn = _make_single_pass_gate_hook(vec_tensor, probe_info, alpha_max)
        handle = _find_transformer_layer(model, layer_idx).register_forward_hook(hook_fn)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=gen_cfg["max_new_tokens"],
                                     temperature=gen_cfg["temperature"],
                                     do_sample=gen_cfg["do_sample"],
                                     pad_token_id=tokenizer.eos_token_id)
        handle.remove()

        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append({**sample, "generated_output": answer, "mode": "single_pass_hard_gate"})
    return results

# ============================================================================
# VERDICT
# ============================================================================

def compute_verdict(df):
    base_row = df[df["mode"] == "base"]
    hard_row = df[df["mode"] == "single_pass_hard_gate_a-1.0"]
    rnd_row = df[df["mode"] == "random_single_pass_hard_gate_a-1.0"]
    shf_row = df[df["mode"] == "shuffled_single_pass_hard_gate_a-1.0"]
    ol_row = df[df["mode"] == "steering_a-1.00"]

    if base_row.empty or hard_row.empty:
        return "NO_DATA", "Missing baseline or hard gate results"

    base_c = base_row["correct_answer_rate"].values[0]
    base_h = base_row["hallucination_rate"].values[0]
    hard_c = hard_row["correct_answer_rate"].values[0]
    hard_h = hard_row["hallucination_rate"].values[0]
    ol_h = ol_row["hallucination_rate"].values[0] if not ol_row.empty else hard_h

    rnd_h = rnd_row["hallucination_rate"].values[0] if not rnd_row.empty else 1.0
    shf_h = shf_row["hallucination_rate"].values[0] if not shf_row.empty else 1.0

    delta_c = hard_c - base_c
    beats_controls = hard_h < min(rnd_h, shf_h)

    if abs(delta_c) <= 0.10 and hard_h <= ol_h * 1.1 and beats_controls:
        return "PASS", f"C stable (dC={delta_c:+.3f}), H={hard_h:.3f} beats controls (rnd={rnd_h:.3f}, shf={shf_h:.3f})"
    elif abs(delta_c) <= 0.15 and hard_h < base_h:
        return "PARTIAL", f"Some improvement: H={hard_h:.3f} < base {base_h:.3f}, dC={delta_c:+.3f}"
    else:
        return "FAIL", f"H={hard_h:.3f} (base={base_h:.3f}), C={hard_c:.3f} (base={base_c:.3f}), beats_controls={beats_controls}"

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    _log("=" * 60, log_path)
    _log("Kaggle P1: M3-v6 Cross-Seed / Cross-Layer Validation", log_path)
    _log("=" * 60, log_path)

    # --------------------------------------------------
    # 1. Load model
    # --------------------------------------------------
    _log(f"\n[1/6] Loading model: {MODEL_NAME}", log_path)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, DEVICE, TORCH_DTYPE)
    total_layers = get_model_layer_count(model)
    _log(f"  Total layers: {total_layers}, device: {next(model.parameters()).device}", log_path)
    _log(f"  Model loaded in {time.time() - t0:.0f}s", log_path)

    total_start = time.time()
    all_rows = []
    all_probe_info = {}
    combo_verdicts = []

    for seed in SEEDS:
        _log(f"\n{'='*50}", log_path)
        _log(f"SEED {seed}", log_path)
        _log(f"{'='*50}", log_path)

        random.seed(seed)
        np.random.seed(seed)

        # --------------------------------------------------
        # 2. Build data
        # --------------------------------------------------
        _log(f"\n  [2/6] Building data...", log_path)
        train, test = build_data(seed, TRAIN_SIZE, TEST_SIZE)
        na_train = sum(1 for s in train if s["answerability"] == "answerable")
        na_test = sum(1 for s in test if s["answerability"] == "answerable")
        _log(f"    Train: {na_train}A+{len(train)-na_train}U, Test: {na_test}A+{len(test)-na_test}U", log_path)

        # --------------------------------------------------
        # 3. Baselines
        # --------------------------------------------------
        _log(f"\n  [3/6] Baselines...", log_path)

        _log(f"    Base...", log_path)
        base_res = generate_answers(model, tokenizer, test, mode="base",
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    temperature=GEN_TEMPERATURE, do_sample=GEN_DO_SAMPLE)
        bm = evaluate_outputs(base_res)
        _log(f"      base: H={bm['hallucination_rate']:.3f} C={bm['correct_answer_rate']:.3f}", log_path)

        _log(f"    Prompt-only...", log_path)
        po_res = generate_answers(model, tokenizer, test, mode="prompt_only",
                                  max_new_tokens=MAX_NEW_TOKENS,
                                  temperature=GEN_TEMPERATURE, do_sample=GEN_DO_SAMPLE)
        pom = evaluate_outputs(po_res)
        _log(f"      prompt_only: H={pom['hallucination_rate']:.3f} C={pom['correct_answer_rate']:.3f}", log_path)

        # Store baselines as "layer=-1" rows
        all_rows.append({"seed": seed, "layer": -1, "mode": "base", "alpha": 0.0,
                         "vector_type": "none", **bm})
        all_rows.append({"seed": seed, "layer": -1, "mode": "prompt_only", "alpha": 0.0,
                         "vector_type": "none", **pom})

        # --------------------------------------------------
        # 4. Per-layer pipeline
        # --------------------------------------------------
        for layer_idx in LAYERS:
            lt0 = time.time()
            _log(f"\n  --- LAYER {layer_idx} ---", log_path)

            # 4a. Collect activations from TRAIN data (30A+30U)
            _log(f"    [4a] Collecting activations...", log_path)
            pos_acts, neg_acts = collect_prefill_activations(model, tokenizer, train, layer_idx)
            hidden_dim = pos_acts.shape[1]
            all_vectors = get_all_vectors(pos_acts, neg_acts, hidden_dim)
            steering_v = all_vectors["steering"]
            _log(f"      {pos_acts.shape[0]} pos + {neg_acts.shape[0]} neg, dim={hidden_dim}", log_path)

            # 4b. Open-loop steering
            _log(f"    [4b] Open-loop steering a={ALPHA_MAX:+.1f}...", log_path)
            ol_res, _ = run_generation_with_steering(
                model, tokenizer, test, steering_v, layer_idx, ALPHA_MAX, "steering",
                max_new_tokens=MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE,
                do_sample=GEN_DO_SAMPLE)
            olm = evaluate_outputs(ol_res)
            _log(f"      OL: H={olm['hallucination_rate']:.3f} C={olm['correct_answer_rate']:.3f}", log_path)
            all_rows.append({"seed": seed, "layer": layer_idx,
                             "mode": f"steering_a{ALPHA_MAX:+.2f}",
                             "alpha": ALPHA_MAX, "vector_type": "steering", **olm})

            # 4c. Probe training
            _log(f"    [4c] Training probe...", log_path)
            probe_layer = layer_idx
            X_train, y_train = collect_probe_features(model, tokenizer, train, probe_layer)
            probe_info = train_probe(X_train, y_train, cv_folds=CV_FOLDS)
            pkey = f"s{seed}_l{layer_idx}"
            all_probe_info[pkey] = {"seed": seed, "layer": layer_idx,
                                    "train_acc": probe_info["train_acc"],
                                    "cv_mean": probe_info.get("cv_mean"),
                                    "auc": probe_info.get("auc")}
            _log(f"      Probe: train_acc={probe_info['train_acc']:.4f}, cv={probe_info.get('cv_mean', 'N/A')}", log_path)

            # 4d. Single-pass hard gate (steering vector)
            _log(f"    [4d] Single-pass hard gate (steering)...", log_path)
            gen_cfg = {"max_new_tokens": MAX_NEW_TOKENS, "temperature": GEN_TEMPERATURE,
                       "do_sample": GEN_DO_SAMPLE}
            hard_res = generate_single_pass_hard_gate(
                model, tokenizer, test, steering_v, layer_idx, ALPHA_MAX, probe_info, gen_cfg)
            hm = evaluate_outputs(hard_res)
            _log(f"      real_gate: H={hm['hallucination_rate']:.3f} C={hm['correct_answer_rate']:.3f}", log_path)
            all_rows.append({"seed": seed, "layer": layer_idx,
                             "mode": "single_pass_hard_gate_a-1.0",
                             "alpha": ALPHA_MAX, "vector_type": "steering", **hm})

            # 4e. Single-pass hard gate (random vector)
            _log(f"    [4e] Single-pass hard gate (random)...", log_path)
            rnd_res = generate_single_pass_hard_gate(
                model, tokenizer, test, all_vectors["random"], layer_idx, ALPHA_MAX, probe_info, gen_cfg)
            rm = evaluate_outputs(rnd_res)
            _log(f"      random: H={rm['hallucination_rate']:.3f} C={rm['correct_answer_rate']:.3f}", log_path)
            all_rows.append({"seed": seed, "layer": layer_idx,
                             "mode": "random_single_pass_hard_gate_a-1.0",
                             "alpha": ALPHA_MAX, "vector_type": "random", **rm})

            # 4f. Single-pass hard gate (shuffled vector)
            _log(f"    [4f] Single-pass hard gate (shuffled)...", log_path)
            shf_res = generate_single_pass_hard_gate(
                model, tokenizer, test, all_vectors["shuffled"], layer_idx, ALPHA_MAX, probe_info, gen_cfg)
            sm = evaluate_outputs(shf_res)
            _log(f"      shuffled: H={sm['hallucination_rate']:.3f} C={sm['correct_answer_rate']:.3f}", log_path)
            all_rows.append({"seed": seed, "layer": layer_idx,
                             "mode": "shuffled_single_pass_hard_gate_a-1.0",
                             "alpha": ALPHA_MAX, "vector_type": "shuffled", **sm})

            _log(f"    Layer {layer_idx} done in {time.time() - lt0:.0f}s", log_path)

        # Per-seed verdict
        df_seed = pd.DataFrame([r for r in all_rows if r["seed"] == seed])
        verdict, reason = compute_verdict(df_seed)
        combo_verdicts.append({"seed": seed, "layer": -1, "verdict": verdict, "reason": reason})
        _log(f"\n  SEED {seed} VERDICT: {verdict}", log_path)

    # --------------------------------------------------
    # 5. Consolidated results
    # --------------------------------------------------
    _log(f"\n{'='*60}", log_path)
    _log(f"CONSOLIDATED RESULTS", log_path)
    _log(f"{'='*60}", log_path)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, "metrics_raw.csv")
    df.to_csv(csv_path, index=False)
    _log(f"\nMetrics saved to {csv_path} ({len(df)} rows)", log_path)

    # Per-combo verdicts
    _log(f"\n{'='*60}", log_path)
    _log(f"PER-COMBO VERDICTS", log_path)
    _log(f"{'='*60}", log_path)

    combo_rows = []
    for seed in SEEDS:
        for layer_idx in LAYERS:
            sub = df[(df["seed"] == seed) & (df["layer"].isin([-1, layer_idx]))]
            v, r = compute_verdict(sub)
            combo_rows.append({"seed": seed, "layer": layer_idx, "verdict": v, "reason": r})

            base_h = sub[sub["mode"] == "base"]["hallucination_rate"]
            hard_h = sub[sub["mode"] == "single_pass_hard_gate_a-1.0"]["hallucination_rate"]
            base_h_val = base_h.values[0] if not base_h.empty else float("nan")
            hard_h_val = hard_h.values[0] if not hard_h.empty else float("nan")
            delta_h = hard_h_val - base_h_val

            _log(f"  Seed={seed} Layer={layer_idx}: {v:6s} | "
                 f"H {base_h_val:.3f}→{hard_h_val:.3f} (dH={delta_h:+.3f}) | {r}", log_path)

    combo_df = pd.DataFrame(combo_rows)
    combo_path = os.path.join(OUTPUT_DIR, "combo_verdicts.csv")
    combo_df.to_csv(combo_path, index=False)

    # Probe info
    probe_df = pd.DataFrame(list(all_probe_info.values()))
    probe_path = os.path.join(OUTPUT_DIR, "probe_evaluation.csv")
    probe_df.to_csv(probe_path, index=False)

    # Summary stats
    passes = sum(1 for c in combo_rows if c["verdict"] == "PASS")
    partials = sum(1 for c in combo_rows if c["verdict"] == "PARTIAL")
    fails = sum(1 for c in combo_rows if c["verdict"] == "FAIL")
    total = len(combo_rows)

    _log(f"\n{'='*60}", log_path)
    _log(f"SUMMARY: {passes}/{total} PASS, {partials}/{total} PARTIAL, {fails}/{total} FAIL", log_path)
    _log(f"{'='*60}", log_path)

    # Cross-seed summary
    _log(f"\nCROSS-SEED SUMMARY:", log_path)
    for seed in SEEDS:
        seed_passes = sum(1 for c in combo_rows if c["seed"] == seed and c["verdict"] == "PASS")
        _log(f"  Seed {seed}: {seed_passes}/{len(LAYERS)} pass", log_path)

    # Cross-layer summary
    _log(f"\nCROSS-LAYER SUMMARY:", log_path)
    for layer_idx in LAYERS:
        layer_passes = sum(1 for c in combo_rows if c["layer"] == layer_idx and c["verdict"] == "PASS")
        _log(f"  Layer {layer_idx}: {layer_passes}/{len(SEEDS)} pass", log_path)

    elapsed = time.time() - total_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)
    _log(f"\nAll outputs in: {OUTPUT_DIR}/", log_path)

    # --------------------------------------------------
    # 6. Generate report
    # --------------------------------------------------
    report_path = os.path.join(OUTPUT_DIR, "P1_CROSS_VALIDATION_REPORT.md")
    lines = []
    lines.append("# IC-4 P1: M3-v6 Cross-Seed / Cross-Layer Validation Report")
    lines.append("")
    lines.append(f"- **Model**: {MODEL_NAME}")
    lines.append(f"- **Device**: {DEVICE}")
    lines.append(f"- **Seeds**: {SEEDS}")
    lines.append(f"- **Layers**: {LAYERS}")
    lines.append(f"- **Alpha**: {ALPHA_MAX}")
    lines.append(f"- **Train**: {TRAIN_SIZE}A+{TRAIN_SIZE}U, Test: {TEST_SIZE}A+{TEST_SIZE}U")
    lines.append(f"- **Total time**: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- **PASS**: {passes}/{total} ({100*passes/total:.0f}%)")
    lines.append(f"- **PARTIAL**: {partials}/{total} ({100*partials/total:.0f}%)")
    lines.append(f"- **FAIL**: {fails}/{total} ({100*fails/total:.0f}%)")
    lines.append("")

    lines.append("## Per-Combo Verdicts")
    lines.append("")
    lines.append("| Seed | Layer | Verdict | Base H | Gate H | dH | Reason |")
    lines.append("|------|-------|---------|--------|--------|-----|--------|")
    for seed in SEEDS:
        for layer_idx in LAYERS:
            sub = df[(df["seed"] == seed) & (df["layer"].isin([-1, layer_idx]))]
            base_h = sub[sub["mode"] == "base"]["hallucination_rate"].values
            gate_h = sub[sub["mode"] == "single_pass_hard_gate_a-1.0"]["hallucination_rate"].values
            bv = base_h[0] if len(base_h) > 0 else float("nan")
            gv = gate_h[0] if len(gate_h) > 0 else float("nan")
            cr = [c for c in combo_rows if c["seed"] == seed and c["layer"] == layer_idx]
            v = cr[0]["verdict"] if cr else "?"
            r = cr[0]["reason"][:60] if cr else ""
            lines.append(f"| {seed} | {layer_idx} | **{v}** | {bv:.3f} | {gv:.3f} | {gv-bv:+.3f} | {r} |")

    lines.append("")
    lines.append("## Probe Performance")
    lines.append("")
    lines.append("| Seed | Layer | Train Acc | CV Mean | AUC |")
    lines.append("|------|-------|-----------|---------|-----|")
    for pinfo in all_probe_info.values():
        lines.append(f"| {pinfo['seed']} | {pinfo['layer']} | {pinfo['train_acc']:.4f} | {pinfo.get('cv_mean', 'N/A')} | {pinfo.get('auc', 'N/A')} |")

    lines.append("")
    lines.append("*Generated by kaggle_p1_cross_validation.py*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _log(f"\nReport saved to {report_path}", log_path)

    return df, combo_rows, probe_df

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df, combos, probes = run_pipeline()
    print("\nDone! Check kaggle_p1_results/ for outputs.")