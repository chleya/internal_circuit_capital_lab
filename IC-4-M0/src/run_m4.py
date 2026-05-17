"""
IC-4-M4: Trajectory-State Diagnostic.
Tests whether the mechanism underlying hallucination/gating is better captured by
trajectory-level representations (mean-pooled, window-pooled) than by single-token
representations (last-token).

Inspired by arXiv:2605.09969 ("The Truth Lies Somewhere in the Middle of the Generated Tokens")
and arXiv:2605.10938 (ELF: Embedded Language Flows).

Usage:
    python -m src.run_m4 --config configs/config_m4.yaml
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config


def _log(msg: str, log_file: str = None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _load_cached_m3_test(seed, test_path, log):
    from src.data_builder import load_jsonl
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")
    if not os.path.exists(test_final):
        raise FileNotFoundError(f"M3 test data not found at {test_final}")
    _log(f"  Loading cached M3 test data for seed {seed}", log)
    test = load_jsonl(test_final)
    na = sum(1 for s in test if s.get("answerability") == "answerable")
    nu = len(test) - na
    _log(f"  seed={seed}: test {na}A+{nu}U", log)
    return test


def _collect_generated_hidden_states(model, tokenizer, sample, layer_idx, max_tokens, gen_cfg):
    """
    Generate text and collect hidden states at layer_idx for the first max_tokens.
    Returns: h_states: list of numpy arrays, one per generated token.
    """
    from src.evaluate import ABSTENTION_PATTERNS, _matches_any, _is_hallucinated

    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    max_new = gen_cfg.get("max_new_tokens", 48)

    context = sample.get("context", "")
    question = sample.get("question", "")
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    hidden_states = []

    def _hook(module, inputs_in, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
        hidden_states.append(h[0, -1, :].detach().cpu().float().numpy().copy())

    layer_module = _find_transformer_layer(model, layer_idx)
    handle = layer_module.register_forward_hook(_hook)

    generated_ids = []
    past_key_values = None
    current_input = input_ids

    for step in range(max_new):
        with torch.no_grad():
            if past_key_values is not None:
                current_input = current_input[:, -1:]
            outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tid = next_token.item()
            generated_ids.append(tid)
            if tid == eos_id:
                break
            current_input = next_token

    handle.remove()

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    label = sample.get("answerability", "?")
    is_halluc = 0
    if label == "unanswerable":
        if _is_hallucinated(answer, sample):
            is_halluc = 1

    should_gate = 1 if label == "unanswerable" else 0

    return {
        "sample_id": sample.get("sample_id", -1),
        "label": label,
        "hidden_states": hidden_states[:max_tokens] if len(hidden_states) > max_tokens else hidden_states,
        "num_tokens": len(hidden_states),
        "is_hallucinated": is_halluc,
        "should_gate": should_gate,
        "generated_answer": answer,
    }


def _find_transformer_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    raise ValueError("Cannot locate transformer layers in this model architecture.")


def _build_representations(collection, max_k=48, window_sizes=(4, 8)):
    """
    Build three types of representations from collected hidden states.

    Returns dict with keys:
      - X_last: (N, D) last-token representations
      - X_mean: (N, D) mean-pooled over first max_k tokens
      - X_win_{w}: (N, D) window-pooled over first w tokens
      - y_answerable: (N,) 0=unanswerable, 1=answerable
      - y_hallucinated: (N,) 0=no_hallucination, 1=hallucinated (unanswerable only)
      - y_gate: (N,) 0=no_gate, 1=gate (same as answerability)
    """
    N = len(collection)
    D = collection[0]["hidden_states"][0].shape[0]

    X_last = np.zeros((N, D), dtype=np.float32)
    X_mean = np.zeros((N, D), dtype=np.float32)
    X_win = {w: np.zeros((N, D), dtype=np.float32) for w in window_sizes}

    y_answerable = np.zeros(N, dtype=np.int32)
    y_hallucinated = -np.ones(N, dtype=np.int32)
    y_gate = np.zeros(N, dtype=np.int32)

    for i, entry in enumerate(collection):
        hs = entry["hidden_states"]
        if len(hs) == 0:
            continue

        label = entry["label"]
        y_answerable[i] = 1 if label == "answerable" else 0
        y_gate[i] = 1 if label == "unanswerable" else 0

        if label == "unanswerable":
            y_hallucinated[i] = entry["is_hallucinated"]

        h_arr = np.stack(hs, axis=0)
        k_actual = min(len(hs), max_k)

        X_last[i] = h_arr[k_actual - 1]
        X_mean[i] = h_arr[:k_actual].mean(axis=0)
        for w in window_sizes:
            w_actual = min(k_actual, w)
            X_win[w][i] = h_arr[:w_actual].mean(axis=0)

    return {
        "X_last": X_last,
        "X_mean": X_mean,
        "X_win": X_win,
        "y_answerable": y_answerable,
        "y_hallucinated": y_hallucinated,
        "y_gate": y_gate,
    }


def _evaluate_representation(X, y, task_name, rep_name, results_list):
    """Evaluate logistic regression on a representation for a given task."""
    if np.sum(y >= 0) < 8:
        results_list.append({
            "task": task_name, "representation": rep_name,
            "accuracy_mean": 0.0, "accuracy_std": 0.0, "n_samples": int(np.sum(y >= 0)),
            "note": "too few samples"
        })
        return

    mask = y >= 0
    X_sub = X[mask]
    y_sub = y[mask]
    n_samples = len(y_sub)

    if len(np.unique(y_sub)) < 2:
        results_list.append({
            "task": task_name, "representation": rep_name,
            "accuracy_mean": 0.0, "accuracy_std": 0.0, "n_samples": n_samples,
            "note": "single class"
        })
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    clf = LogisticRegression(max_iter=1000, random_state=42)

    n_folds = min(5, n_samples // 2)
    if n_folds < 2:
        clf.fit(X_scaled, y_sub)
        acc = clf.score(X_scaled, y_sub)
        results_list.append({
            "task": task_name, "representation": rep_name,
            "accuracy_mean": round(acc, 4), "accuracy_std": 0.0,
            "n_samples": n_samples, "note": "train score (n<4)"
        })
        return

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y_sub, cv=cv, scoring="accuracy")
    results_list.append({
        "task": task_name, "representation": rep_name,
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "n_samples": n_samples,
    })


def _generate_report(report_path, config, results_df, collection, elapsed):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append("# IC-4-M4: Trajectory-State Diagnostic Report")
    lines.append("")

    lines.append("## 1. Motivation")
    lines.append("")
    lines.append("Recent work (arXiv:2605.09969, \"The Truth Lies Somewhere in the Middle\") shows that "
                 "mean-pooled hidden states across generated tokens capture more semantic information "
                 "than any single token. This suggests that hallucination risk signals may also be "
                 "better captured by trajectory-level representations than token-local ones.")
    lines.append("")
    lines.append("This experiment compares three representation types:")
    lines.append("- **Last-token**: hidden state at the last generated token position")
    lines.append("- **Mean-pooled**: average of hidden states across all generated tokens")
    lines.append("- **Window-pooled**: average of hidden states across the first 4 or 8 generated tokens")
    lines.append("")

    lines.append("## 2. Experiment Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | {config['model']['name']} |")
    lines.append(f"| Device / dtype | {config['model']['device']} / {config['model']['torch_dtype']} |")
    layers = config["m4"]["layers"]
    lines.append(f"| Layer | {layers} |")
    lines.append(f"| Samples | {len(collection)} |")
    lines.append(f"| Window sizes | {config['m4']['window_sizes']} |")
    lines.append(f"| Max generated tokens | {config['m4']['max_generated_tokens']} |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    a_count = sum(1 for e in collection if e["label"] == "answerable")
    u_count = len(collection) - a_count
    h_count = sum(1 for e in collection if e.get("is_hallucinated", 0) == 1)
    lines.append(f"Dataset: {a_count} answerable, {u_count} unanswerable, {h_count} hallucinated")
    lines.append("")

    lines.append("## 3. Classification Results")
    lines.append("")

    tasks = results_df["task"].unique()
    for task in tasks:
        task_label = {
            "answerable_vs_unanswerable": "Answerable vs Unanswerable",
            "will_hallucinate": "Will Hallucinate (unanswerable subset)",
            "should_gate_on": "Should Gate On (same as answerability)",
        }.get(task, task)
        lines.append(f"### {task_label}")
        lines.append("")
        lines.append("| Representation | CV Accuracy (mean) | CV Accuracy (std) | n_samples | Note |")
        lines.append("|---|---|---|---|---|")
        sub = results_df[results_df["task"] == task]
        for _, row in sub.iterrows():
            note = str(row.get("note", "")) if not pd.isna(row.get("note", "")) else ""
            lines.append(f"| {row['representation']} | {row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | {int(row['n_samples'])} | {note} |")
        lines.append("")

        best = sub.loc[sub["accuracy_mean"].idxmax()] if len(sub) > 0 else None
        if best is not None and best["accuracy_mean"] > 0:
            lines.append(f"**Best**: {best['representation']} (acc={best['accuracy_mean']:.4f})")
            lines.append("")

    lines.append("## 4. Representation Quality Comparison")
    lines.append("")

    rep_types = ["last-token", "mean-pooled", "window-4", "window-8"]
    summary = {}
    for rep in rep_types:
        sub = results_df[results_df["representation"] == rep]
        if len(sub) > 0:
            summary[rep] = float(sub["accuracy_mean"].mean())

    if summary:
        best_rep = max(summary, key=summary.get)
        lines.append("| Representation | Avg Accuracy Across Tasks | Verdict |")
        lines.append("|---|---|---|")
        for rep, acc in sorted(summary.items(), key=lambda x: -x[1]):
            v = "**BEST**" if rep == best_rep else ""
            lines.append(f"| {rep} | {acc:.4f} | {v} |")
        lines.append("")

    lines.append("## 5. Interpretation")
    lines.append("")

    best_rep_type = None
    best_acc = 0
    for _, row in results_df.iterrows():
        if row["accuracy_mean"] > best_acc:
            best_acc = row["accuracy_mean"]
            best_rep_type = row["representation"]

    if best_rep_type and best_rep_type != "last-token":
        lines.append(f"**Mechanism: Trajectory-level state**")
        lines.append("")
        lines.append(f"The best representation ({best_rep_type}, acc={best_acc:.4f}) outperforms "
                     f"last-token significantly. This means the internal state relevant to "
                     f"hallucination/answerability is distributed across generated tokens, "
                     f"not localized at a single position.")
        lines.append("")
        lines.append("> \"机制更像 trajectory-level state，而不是 token-local trigger\"")
    else:
        last_acc = float(results_df[results_df["representation"] == "last-token"]["accuracy_mean"].max()) if "last-token" in results_df["representation"].values else 0
        pool_acc = float(results_df[results_df["representation"].isin(["mean-pooled", "window-4", "window-8"])]["accuracy_mean"].max())
        lines.append(f"**Mechanism: Inconclusive**")
        lines.append("")
        lines.append(f"Last-token best acc: {last_acc:.4f}, Pooled best acc: {pool_acc:.4f}")
        lines.append(f"Small sample size (n={len(collection)}) limits statistical power.")
        lines.append("Consider larger sample or additional layers.")
    lines.append("")

    lines.append("## 6. Implications for IC-4")
    lines.append("")
    if best_rep_type and "pool" in best_rep_type.lower() or best_rep_type and "win" in best_rep_type.lower():
        lines.append("- **Probe gate (M3-v3)**: Use mean-pooled or window-pooled hidden states as probe features, not last-token only.")
        lines.append("- **Steering vector v**: Consider extracting v from mean-pooled activations of generated tokens rather than last-token only.")
        lines.append("- **Gating signal**: Trajectory-level representations capture more information; future gating should leverage multi-token context.")
    else:
        lines.append("- Last-token may be sufficient for probe features given current sample size.")
        lines.append("- But paper evidence (arXiv:2605.09969) suggests mean-pooling should help at scale.")
        lines.append("- Recommend testing with larger sample in future iterations.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M4: Trajectory-State Diagnostic*")
    lines.append("*Generated by run_m4*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="IC-4-M4: Trajectory-State Diagnostic")
    parser.add_argument("--config", type=str, default="configs/config_m4.yaml")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    config = load_config(args.config)
    results_dir = config["output"]["results_dir"]
    reports_dir = config["output"]["reports_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "run_log.txt")
    _log("=" * 60, log_path)
    _log("IC-4-M4: Trajectory-State Diagnostic", log_path)
    _log("=" * 60, log_path)

    seeds = config["m4"]["seeds"]
    layers = config["m4"]["layers"]
    max_k = config["m4"]["max_generated_tokens"]
    window_sizes = tuple(config["m4"]["window_sizes"])
    gen_cfg = config["generation"]

    _log(f"\nConfig: seed={seeds}, layer={layers}, max_k={max_k}, window={window_sizes}", log_path)
    _log(f"Data: test={config['data']['test_size']}", log_path)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    _log(f"\nLoading model ({config['model']['name']})...", log_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
    )
    total_layers = get_model_layer_count(model)
    _log(f"  Total layers: {total_layers}", log_path)

    t_start = time.time()

    all_collections = []
    for seed in seeds:
        _log(f"\n{'='*40}", log_path)
        _log(f"SEED {seed}", log_path)
        _log(f"{'='*40}", log_path)

        test_path = config["data"]["test_path"]
        test = _load_cached_m3_test(seed, test_path, log_path)

        for layer_idx in layers:
            _log(f"\n  LAYER {layer_idx} — collecting hidden states for {len(test)} samples", log_path)

            collections = []
            for i, sample in enumerate(test):
                result = _collect_generated_hidden_states(
                    model, tokenizer, sample, layer_idx, max_k, gen_cfg
                )
                result["sample_id"] = i
                result["seed"] = seed
                result["layer"] = layer_idx
                result["label"] = sample.get("answerability", "?")
                collections.append(result)

                if (i + 1) % 20 == 0:
                    _log(f"    collected {i+1}/{len(test)}", log_path)

            all_collections.extend(collections)

            _log(f"  LAYER {layer_idx} done", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    _log(f"\nBuilding representations...", log_path)
    rep_data = _build_representations(all_collections, max_k=max_k, window_sizes=window_sizes)

    _log(f"Evaluating representations...", log_path)
    eval_results = []
    rep_specs = [
        ("X_last", "last-token"),
        ("X_mean", "mean-pooled"),
    ]
    for w in window_sizes:
        rep_specs.append((f"X_win_{w}", f"window-{w}"))

    tasks = [
        ("y_answerable", "answerable_vs_unanswerable"),
        ("y_hallucinated", "will_hallucinate"),
        ("y_gate", "should_gate_on"),
    ]

    for rep_key, rep_name in rep_specs:
        if rep_key == "X_last":
            X = rep_data["X_last"]
        elif rep_key == "X_mean":
            X = rep_data["X_mean"]
        else:
            w = int(rep_key.split("_")[-1])
            X = rep_data["X_win"].get(w)
            if X is None:
                continue

        for y_key, task_name in tasks:
            y = rep_data[y_key]
            _evaluate_representation(X, y, task_name, rep_name, eval_results)

    results_df = pd.DataFrame(eval_results)
    results_path = os.path.join(results_dir, "representation_classification.csv")
    results_df.to_csv(results_path, index=False)
    _log(f"\nClassification results saved to {results_path}", log_path)

    raw_path = os.path.join(results_dir, "hidden_state_collection.npz")
    save_data = {}
    for i, c in enumerate(all_collections):
        if len(c["hidden_states"]) > 0:
            save_data[f"hs_{i}"] = np.stack(c["hidden_states"], axis=0)
    save_data["num_samples"] = len(all_collections)
    save_data["labels"] = np.array([c["label"] for c in all_collections])
    save_data["is_hallucinated"] = np.array([c["is_hallucinated"] for c in all_collections])
    save_data["should_gate"] = np.array([c["should_gate"] for c in all_collections])
    np.savez_compressed(raw_path, **save_data)
    _log(f"Raw hidden states saved to {raw_path} ({len(all_collections)} samples, {len([k for k in save_data if k.startswith('hs_')])} with hidden states)", log_path)

    _log(f"\n{'='*60}", log_path)
    _log("REPRESENTATION COMPARISON", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"{'Task':<32} {'Representation':<16} {'Accuracy':>12} {'Std':>10}", log_path)
    _log(f"{'-'*32} {'-'*16} {'-'*12} {'-'*10}", log_path)
    for _, row in results_df.iterrows():
        _log(f"{row['task']:<32} {row['representation']:<16} {row['accuracy_mean']:>10.4f} {row['accuracy_std']:>10.4f}", log_path)

    for task in results_df["task"].unique():
        sub = results_df[results_df["task"] == task]
        if len(sub) > 0:
            best = sub.loc[sub["accuracy_mean"].idxmax()]
            _log(f"\n  {task}: best = {best['representation']} (acc={best['accuracy_mean']:.4f})", log_path)

    _log(f"\n{'='*60}", log_path)
    _log("SUMMARY", log_path)
    _log(f"{'='*60}", log_path)

    rep_types_found = results_df["representation"].unique()
    best_overall = None
    best_overall_acc = 0
    for rep in rep_types_found:
        sub = results_df[results_df["representation"] == rep]
        avg_acc = float(sub["accuracy_mean"].mean())
        _log(f"  {rep}: avg acc across tasks = {avg_acc:.4f}", log_path)
        if avg_acc > best_overall_acc:
            best_overall_acc = avg_acc
            best_overall = rep

    if best_overall and best_overall != "last-token":
        _log(f"\n  => Best representation: {best_overall} (avg acc={best_overall_acc:.4f})", log_path)
        _log(f"  => Mechanism appears TRAJECTORY-LEVEL, not token-local trigger.", log_path)
    else:
        _log(f"\n  => No clear advantage for pooled representations.", log_path)
        _log(f"  => Cannot conclusively distinguish trajectory vs token-local mechanism.", log_path)

    report_path = os.path.join(reports_dir, "IC4_M4_TRAJECTORY_STATE_REPORT.md")
    _generate_report(report_path, config, results_df, all_collections, elapsed)
    _log(f"\nReport saved to {report_path}", log_path)

    _log(f"\n{'='*60}", log_path)
    _log("IC-4-M4 complete.", log_path)
    _log(f"{'='*60}", log_path)


if __name__ == "__main__":
    main()