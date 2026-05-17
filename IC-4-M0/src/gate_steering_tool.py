"""
IC-4-M0: GateSteeringTool — Reusable Reference Implementation.

Encapsulates the M3-v6 proven mechanism:
  last_prompt_token + logistic probe + hard threshold
  + single-pass forward hook + model.generate()

Design principles:
  - No manual token-by-token generation (use model.generate() exclusively)
  - No open-loop sweeps (only gated steering)
  - Reserved entry points for cross-seed / cross-layer validation
  - Self-contained: import, configure, train_probe, generate

Verified against M3-v6: H=0.667, C=0.600, UA=0.000 (oracle-level).
"""

import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


class GateSteeringTool:
    """
    Single-pass hook-based hard gate for conditional activation steering.

    Parameters
    ----------
    model : PreTrainedModel
        Loaded HuggingFace transformer model.
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    config : dict
        Configuration with keys: threshold (float), cv_folds (int).
    """

    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        cfg = config or {}
        self.threshold = cfg.get("threshold", 0.5)
        self.cv_folds = cfg.get("cv_folds", 3)
        self.max_new_tokens = cfg.get("max_new_tokens", 48)
        self.temperature = cfg.get("temperature", 0.0)
        self.do_sample = cfg.get("do_sample", False)

        self._probe_info = None
        self._layer_idx = None
        self._representation = None

    # =========================================================================
    # Probe Training
    # =========================================================================

    def train_probe(self, train_data, layer_idx, representation="last_prompt_token"):
        """
        Train a logistic probe on prefill hidden states.

        Parameters
        ----------
        train_data : list[dict]
            Training samples with 'context', 'question', 'answerability' keys.
        layer_idx : int
            Transformer layer index to extract hidden states from.
        representation : str
            Pooling method: 'last_prompt_token' or 'mean_pooled'.

        Returns
        -------
        probe_info : dict
            Contains classifier, scaler, train_acc, cv_acc_mean, auc, n_samples.
        """
        self._layer_idx = layer_idx
        self._representation = representation

        X_list, y_list = [], []
        for sample in train_data:
            prompt = self._build_prompt(sample)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            hs = outputs.hidden_states[layer_idx + 1][0]

            if representation == "last_prompt_token":
                pooled = hs[-1, :].detach().cpu().float().numpy()
            elif representation == "mean_pooled":
                pooled = hs.mean(dim=0).detach().cpu().float().numpy()
            else:
                pooled = hs[-1, :].detach().cpu().float().numpy()

            X_list.append(pooled)
            y_list.append(1 if sample.get("answerability") == "answerable" else 0)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_scaled, y)

        train_acc = accuracy_score(y, clf.predict(X_scaled))

        n_folds = min(self.cv_folds, len(y) // 2)
        cv_scores = []
        if n_folds >= 2 and len(np.unique(y)) >= 2:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")

        auc_score = None
        if len(np.unique(y)) >= 2:
            try:
                auc_score = roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1])
            except Exception:
                pass

        self._probe_info = {
            "classifier": clf,
            "scaler": scaler,
            "train_acc": float(train_acc),
            "cv_acc_mean": float(np.mean(cv_scores)) if len(cv_scores) > 0 else None,
            "auc": float(auc_score) if auc_score is not None else None,
            "n_samples": len(y),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
            "layer_idx": layer_idx,
            "representation": representation,
        }
        return self._probe_info

    # =========================================================================
    # Probe Evaluation on External Data
    # =========================================================================

    def evaluate_probe(self, test_data, probe_info=None):
        """
        Evaluate probe accuracy on a held-out dataset by extracting hidden states.

        Parameters
        ----------
        test_data : list[dict]
            Test samples with 'context', 'question', 'answerability'.
        probe_info : dict, optional
            Probe info. Uses self._probe_info if None.

        Returns
        -------
        dict
            accuracy, n_samples, n_correct, y_true, y_pred, proba.
        """
        info = probe_info or self._probe_info
        if info is None:
            raise ValueError("No probe trained.")

        representation = info.get("representation", self._representation or "last_prompt_token")
        layer_idx = info.get("layer_idx", self._layer_idx)

        X_list, y_list = [], []
        for sample in test_data:
            prompt = self._build_prompt(sample)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            hs = outputs.hidden_states[layer_idx + 1][0]
            if representation == "last_prompt_token":
                pooled = hs[-1, :].detach().cpu().float().numpy()
            elif representation == "mean_pooled":
                pooled = hs.mean(dim=0).detach().cpu().float().numpy()
            else:
                pooled = hs[-1, :].detach().cpu().float().numpy()
            X_list.append(pooled)
            y_list.append(1 if sample.get("answerability") == "answerable" else 0)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int32)
        X_scaled = info["scaler"].transform(X)
        proba = info["classifier"].predict_proba(X_scaled)[:, 1]
        y_pred = info["classifier"].predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        return {
            "accuracy": float(acc),
            "n_samples": len(y),
            "n_correct": int(np.sum(y_pred == y)),
            "y_true": y.tolist(),
            "y_pred": y_pred.tolist(),
            "proba": proba.tolist(),
        }

    # =========================================================================
    # Generation (Single Sample)
    # =========================================================================

    def generate(self, sample, steering_vector, layer_idx, alpha, probe_info=None,
                 gate_type="hard", soft_temperature=0.1, confidence_zone=(0.35, 0.65)):
        """
        Run single-pass hook-based gate generation on one sample.

        Parameters
        ----------
        sample : dict
            Test sample with 'context', 'question', 'answerability'.
        steering_vector : np.ndarray
            Steering vector (shape: [hidden_dim]).
        layer_idx : int
            Transformer layer index for hook registration.
        alpha : float
            Steering strength (negative to suppress hallucination).
        probe_info : dict, optional
            Pre-trained probe info. Uses self._probe_info if None.
        gate_type : str
            "hard": probe_score >= threshold → no steer, else full alpha.
            "soft": effective_alpha = alpha * sigmoid((threshold - score) / T).
            "confidence_aware": skip if score in confidence_zone, else hard gate.
        soft_temperature : float
            Temperature for soft gate sigmoid (lower = sharper).
        confidence_zone : tuple (low, high)
            Ambiguity zone for confidence-aware gate.

        Returns
        -------
        dict
            Sample fields + generated_output, probe_score, gate, alpha_applied, gate_type.
        """
        info = probe_info or self._probe_info
        if info is None:
            raise ValueError("No probe trained. Call train_probe() first or pass probe_info.")

        representation = info.get("representation", self._representation or "last_prompt_token")
        threshold = self.threshold
        vec_tensor = torch.from_numpy(np.asarray(steering_vector)).to(self.device).float()
        layer_module = self._find_layer(layer_idx)

        prompt = self._build_prompt(sample)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gate_decided = [False]
        probe_score = [0.5]
        gate_val = [0.0]
        effective_alpha = [0.0]

        def make_hook():
            def hook(module, fn_inputs, fn_outputs):
                if isinstance(fn_outputs, tuple):
                    h_full = fn_outputs[0]
                else:
                    h_full = fn_outputs

                if not gate_decided[0]:
                    h = h_full[0] if h_full.dim() == 3 else h_full

                    if representation == "last_prompt_token":
                        pooled = h[-1, :].detach().cpu().float().numpy()
                    elif representation == "mean_pooled":
                        pooled = h.mean(dim=0).detach().cpu().float().numpy()
                    else:
                        pooled = h[-1, :].detach().cpu().float().numpy()

                    X = info["scaler"].transform(pooled.reshape(1, -1))
                    proba = info["classifier"].predict_proba(X)[0, 1]
                    probe_score[0] = float(proba)
                    gate_decided[0] = True

                    if gate_type == "hard":
                        if probe_score[0] >= threshold:
                            gate_val[0] = 0.0
                            effective_alpha[0] = 0.0
                        else:
                            gate_val[0] = 1.0
                            effective_alpha[0] = alpha

                    elif gate_type == "soft":
                        raw = (threshold - probe_score[0]) / max(soft_temperature, 0.001)
                        sigmoid_val = 1.0 / (1.0 + np.exp(-raw))
                        gate_val[0] = float(sigmoid_val)
                        effective_alpha[0] = alpha * gate_val[0]

                    elif gate_type == "confidence_aware":
                        lo, hi = confidence_zone
                        if lo < probe_score[0] < hi:
                            gate_val[0] = 0.0
                            effective_alpha[0] = 0.0
                        elif probe_score[0] >= threshold:
                            gate_val[0] = 0.0
                            effective_alpha[0] = 0.0
                        else:
                            gate_val[0] = 1.0
                            effective_alpha[0] = alpha

                    else:
                        if probe_score[0] >= threshold:
                            gate_val[0] = 0.0
                            effective_alpha[0] = 0.0
                        else:
                            gate_val[0] = 1.0
                            effective_alpha[0] = alpha

                if abs(effective_alpha[0]) > 0.001:
                    v = vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
                    h_modified = h_full + effective_alpha[0] * v
                    if isinstance(fn_outputs, tuple):
                        return (h_modified,) + fn_outputs[1:]
                    else:
                        return h_modified
                else:
                    return None

            return hook

        handle = layer_module.register_forward_hook(make_hook())

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        handle.remove()

        answer = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        return {
            **sample,
            "generated_output": answer,
            "probe_score": round(probe_score[0], 6),
            "gate": gate_val[0],
            "alpha_applied": effective_alpha[0],
            "gate_type": gate_type,
        }

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def generate_batch(self, test_data, steering_vector, layer_idx, alpha,
                       probe_info=None, control_type="steering",
                       gate_type="hard", soft_temperature=0.1, confidence_zone=(0.35, 0.65)):
        """
        Run gate generation on a batch of test samples.

        Parameters
        ----------
        test_data : list[dict]
            Test samples.
        steering_vector : np.ndarray
            Steering vector.
        layer_idx : int
            Transformer layer index.
        alpha : float
            Steering strength.
        probe_info : dict, optional
            Pre-trained probe info.
        control_type : str
            Label for the steering vector type ('steering', 'random', 'shuffled').
        gate_type : str
            "hard", "soft", or "confidence_aware".
        soft_temperature : float
            Temperature for soft gate.
        confidence_zone : tuple (low, high)
            Ambiguity zone for confidence-aware gate.

        Returns
        -------
        list[dict]
            Results with generated_output, probe_score, gate, alpha_applied, gate_type.
        """
        info = probe_info or self._probe_info
        results = []
        for sample in test_data:
            result = self.generate(sample, steering_vector, layer_idx, alpha,
                                   probe_info=info, gate_type=gate_type,
                                   soft_temperature=soft_temperature,
                                   confidence_zone=confidence_zone)
            result["control_type"] = control_type
            result["layer_idx"] = layer_idx
            result["alpha"] = alpha
            results.append(result)
        return results

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate(self, results):
        """
        Compute H, C, UA metrics from generation results.

        Parameters
        ----------
        results : list[dict]
            Output from generate_batch(), with 'generated_output' and 'answerability'.

        Returns
        -------
        dict
            Metrics: hallucination_rate, correct_answer_rate, unnecessary_abstention_rate,
            answerable_count, unanswerable_count.
        """
        from src.evaluate import evaluate_outputs
        return evaluate_outputs(results)

    # =========================================================================
    # Reserved Entry Points (cross-seed / cross-layer)
    # =========================================================================

    def set_layer(self, layer_idx):
        """
        RESERVED: Switch to a different steering layer.
        Requires corresponding activation file at
        results_m3/activations_s{seed}_l{layer_idx}.npz
        """
        self._layer_idx = layer_idx

    def set_representation(self, representation):
        """
        RESERVED: Switch probe representation type.
        Supported: 'last_prompt_token', 'mean_pooled'.
        """
        self._representation = representation

    @staticmethod
    def load_activation(seed, layer_idx, results_dir="results_m3"):
        """
        RESERVED: Load pre-computed steering vectors for a given seed/layer.
        Returns dict with 'steering', 'random', 'shuffled' keys.
        """
        from src.activation_collector import load_activations
        from src.steering import get_all_vectors
        act_path = os.path.join(results_dir, f"activations_s{seed}_l{layer_idx}.npz")
        acts = load_activations(act_path)
        hidden_dim = acts["positive"].shape[1]
        return get_all_vectors(acts["positive"], acts["negative"], hidden_dim)

    # =========================================================================
    # Internal
    # =========================================================================

    def _build_prompt(self, sample):
        context = sample.get("context", "")
        question = sample.get("question", "")
        return f"{context}\n\nQuestion: {question}\nAnswer:"

    def _find_layer(self, layer_idx):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
            return self.model.model.decoder.layers[layer_idx]
        raise ValueError("Cannot locate transformer layers in model.")