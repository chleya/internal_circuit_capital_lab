"""
Phase 9-B: Multi-Checkpoint LLM Consolidation Data Generation.
===============================================================
Fixes the Phase 7 3.3B data-gap: generates hidden states from Qwen-0.5B
at DIFFERENT fine-tuning checkpoints, then tests PerClassKMeans consolidation.

Background:
  Phase 7 3.3B failed because all M3 activations came from the same base
  Qwen model — seeds split from the same distribution produce identical
  representations (KNN=1.0 at 2D PCA). The genuine test requires hidden
  states from DIFFERENT model snapshots.

Method:
  1. LoRA fine-tune Qwen-0.5B on M3 training data
  2. Save checkpoints every epoch (5 checkpoints)
  3. Extract hidden states at layer 12 from each checkpoint
  4. Treat different checkpoints as "states to consolidate"
  5. Test: PerClassKMeans (binary version) on checkpoint-cross data

Hypotheses:
  H9.3: Multi-checkpoint hidden states show divergence (KNN < 1.0)
  H9.4: PerClassKMeans outperforms XOnlyKMeans across checkpoints
  H9.5: YAwareKMeans provides intermediate advantage

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_c7_multi_checkpoint_consolidation.py --epochs 5 --rank 4
"""

import argparse
import os, sys, time, json, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import evaluate_outputs
from src.run_m3_v6 import _collect_prefill_features
from src.run_m2 import load_config

RESULTS_DIR = "results_c7_multi_checkpoint"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

class M3Dataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        full_text = f"{prompt} {answer}"

        inputs = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        prompt_len = prompt_tokens["input_ids"].shape[1]
        labels[0, :prompt_len] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    log(f"  Epoch {epoch}: loss={avg_loss:.4f}, steps={n_batches}")
    return avg_loss

def extract_features_from_checkpoint(model, tokenizer, samples, layer, repr_type):
    model.eval()
    X_list = []
    device = next(model.parameters()).device

    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states[layer]
            last_token_hs = hs[0, -1, :].cpu().numpy()
            X_list.append(last_token_hs)

    return np.array(X_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    parser.add_argument("--skip_training", action="store_true", help="Skip training and only run analysis on saved checkpoints")

    log("=" * 64)
    log("Phase 9-B: Multi-Checkpoint LLM Consolidation")
    log(f"  epochs={args.epochs}, rank={args.rank}, lr={args.lr}")
    if args.skip_training:
        log("  MODE: Analysis only (skipping training)")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading base model on CPU...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        torch_dtype="float32",
    )
    device = next(model.parameters()).device
    log(f"  Model on {device}.")

    SEED = 0
    LAYER = 12
    REPR = "last_prompt_token"

    log(f"\n[Step 2] Loading M3 data...")
    config = load_config("configs/config_m3_v6.yaml")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = config.get("data_dir", "data_m3")
    train_path = os.path.join(base_dir, data_dir, f"train_s{SEED}.jsonl")
    test_path = os.path.join(base_dir, data_dir, f"test_s{SEED}.jsonl")
    train_samples = load_jsonl(train_path)
    test_samples = load_jsonl(test_path)
    log(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

    log(f"\n[Step 3] Extracting BASE features (pre-training)...")
    X_base_train = extract_features_from_checkpoint(model, tokenizer, train_samples, LAYER, REPR)
    X_base_test = extract_features_from_checkpoint(model, tokenizer, test_samples, LAYER, REPR)
    y_train = np.array([1 if s.get("answerability") == "answerable" else 0 for s in train_samples])
    y_test = np.array([1 if s.get("answerability") == "answerable" else 0 for s in test_samples])
    log(f"  X_train: {X_base_train.shape}, X_test: {X_base_test.shape}")

    np.save(os.path.join(RESULTS_DIR, "X_train_checkpoint_0.npy"), X_base_train)
    np.save(os.path.join(RESULTS_DIR, "X_test_checkpoint_0.npy"), X_base_test)
    np.save(os.path.join(RESULTS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)

    log(f"\n[Step 4] Applying LoRA (rank={args.rank})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    log(f"\n[Step 5] Training for {args.epochs} epochs...")
    dataset = M3Dataset(train_samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_features_train = {0: X_base_train}
    checkpoint_features_test = {0: X_base_test}

    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, device, epoch)

        ckpt_dir = os.path.join(RESULTS_DIR, f"checkpoint_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        log(f"  Saved LoRA checkpoint to {ckpt_dir}")

        log(f"  Extracting features from epoch {epoch}...")
        X_train_ep = extract_features_from_checkpoint(model, tokenizer, train_samples, LAYER, REPR)
        X_test_ep = extract_features_from_checkpoint(model, tokenizer, test_samples, LAYER, REPR)
        checkpoint_features_train[epoch] = X_train_ep
        checkpoint_features_test[epoch] = X_test_ep

        np.save(os.path.join(RESULTS_DIR, f"X_train_checkpoint_{epoch}.npy"), X_train_ep)
        np.save(os.path.join(RESULTS_DIR, f"X_test_checkpoint_{epoch}.npy"), X_test_ep)

    train_elapsed = time.time() - t_train
    log(f"  Training + extraction: {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")

    log(f"\n[Step 6] Cross-checkpoint divergence analysis...")
    n_ckpts = len(checkpoint_features_test)
    log(f"  Checkpoints: {list(checkpoint_features_test.keys())}")
    log(f"\n  {'Ckpt Pair':12s} | {'KNN Acc':>8s} | {'||mean diff||':>12s} | {'CosSim':>8s}")
    log(f"  {'-'*52}")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    for i in range(n_ckpts):
        for j in range(i+1, n_ckpts):
            Xi = checkpoint_features_test[i]
            Xj = checkpoint_features_test[j]

            scaler = StandardScaler()
            Xi_scaled = scaler.fit_transform(Xi)
            Xj_scaled = scaler.transform(Xj)

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(Xi_scaled, y_test)
            y_pred = knn.predict(Xj_scaled)
            acc = accuracy_score(y_test, y_pred)

            mean_diff = np.linalg.norm(Xi.mean(axis=0) - Xj.mean(axis=0))

            cos_sims = []
            for k in range(min(len(Xi), len(Xj))):
                cos_sim = np.dot(Xi[k], Xj[k]) / (np.linalg.norm(Xi[k]) * np.linalg.norm(Xj[k]) + 1e-8)
                cos_sims.append(cos_sim)
            avg_cos = np.mean(cos_sims)

            log(f"  {i} vs {j:<8d} | {acc:8.4f} | {mean_diff:12.4f} | {avg_cos:8.4f}")

    log(f"\n[Step 7] PerClassKMeans Consolidation Test...")
    from sklearn.cluster import KMeans

    ckpt_pairs = [(0, args.epochs), (0, args.epochs // 2), (args.epochs // 2, args.epochs)] if args.epochs >= 2 else [(0, 1)]

    log(f"\n  {'Pair':10s} | {'X-Only':>8s} | {'Y-Aware':>8s} | {'PerClass':>8s} | {'Freq':>8s} | {'PC-None':>8s}")
    log(f"  {'-'*65}")

    for (ci, cj) in ckpt_pairs:
        if ci not in checkpoint_features_train or cj not in checkpoint_features_test:
            continue

        Xi_train = checkpoint_features_train[ci]
        Xj_test = checkpoint_features_test[cj]
        yj_test = y_test

        n_samples = len(yj_test)
        n_halfA = sum(1 for y in yj_test if y == 1)
        n_halfU = n_samples - n_halfA

        scaler_full = StandardScaler()
        Xi_scaled = scaler_full.fit_transform(Xi_train)
        Xj_scaled = scaler_full.transform(Xj_test)

        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km_labels = km.fit_predict(Xj_scaled)
        match_0 = max(
            (km_labels == 0).astype(int) == yj_test,
            (km_labels == 1).astype(int) == yj_test
        )
        km_acc = np.mean(1 - km_labels == yj_test) if np.mean(km_labels == yj_test) < 0.5 else np.mean(km_labels == yj_test)
        km_acc = max(np.mean(km_labels == yj_test), np.mean(1 - km_labels == yj_test))

        ya_labels = np.zeros(n_samples, dtype=int)
        ya_labels[:n_halfA] = 0
        ya_labels[n_halfA:] = 1
        np.random.shuffle(ya_labels)
        y_random = np.random.randint(0, 2, n_samples)
        freq_acc = max(np.mean(y_random == yj_test), np.mean(1 - y_random == yj_test))

        centroids_A = Xi_scaled[y_train == 1].mean(axis=0)
        centroids_U = Xi_scaled[y_train == 0].mean(axis=0)
        dist_A = np.linalg.norm(Xj_scaled - centroids_A, axis=1)
        dist_U = np.linalg.norm(Xj_scaled - centroids_U, axis=1)
        ya_labels = (dist_A < dist_U).astype(int)
        ya_acc = max(np.mean(ya_labels == yj_test), np.mean(1 - ya_labels == yj_test))

        pc_labels = np.zeros(n_samples, dtype=int)
        for k in range(n_samples):
            idx_A = np.where(y_train == 1)[0]
            idx_U = np.where(y_train == 0)[0]
            dist_to_A = np.linalg.norm(Xj_scaled[k] - Xi_scaled[idx_A], axis=1).min()
            dist_to_U = np.linalg.norm(Xj_scaled[k] - Xi_scaled[idx_U], axis=1).min()
            pc_labels[k] = 1 if dist_to_A < dist_to_U else 0
        pc_acc = max(np.mean(pc_labels == yj_test), np.mean(1 - pc_labels == yj_test))

        no_mem = max(n_halfA, n_halfU) / n_samples
        log(f"  {ci}→{cj:6s} | {km_acc:8.4f} | {ya_acc:8.4f} | {pc_acc:8.4f} | {freq_acc:8.4f} | {pc_acc-no_mem:+8.4f}")

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log(f"Phase 9-B Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")
    log("=" * 64)

    summary = {
        "epochs": args.epochs,
        "rank": args.rank,
        "lr": args.lr,
        "n_checkpoints": n_ckpts,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nResults saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()