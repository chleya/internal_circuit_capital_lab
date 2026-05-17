"""Quick test: hook active during prefill AND generation."""
import os, sys, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model_loader import load_model_and_tokenizer, get_model_layer_count
from src.activation_collector import load_activations
from src.steering import get_all_vectors
from src.data_builder import load_jsonl

model, tokenizer = load_model_and_tokenizer('Qwen/Qwen2.5-0.5B-Instruct', 'cpu', 'float32')

train = load_jsonl('data_m3/train_s0.jsonl')
test = load_jsonl('data_m3/test_s0.jsonl')

acts = load_activations('results_m3/activations_s0_l12.npz')
vectors = get_all_vectors(acts['positive'], acts['negative'], acts['positive'].shape[1])
steering_v = vectors['steering']

X_list, y_list = [], []
for s in train:
    prompt = f"{s.get('context','')}\n\nQuestion: {s.get('question','')}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    hs = outputs.hidden_states[13][0]
    X_list.append(hs[-1,:].detach().cpu().float().numpy())
    y_list.append(1 if s.get('answerability')=='answerable' else 0)
X = np.stack(X_list, axis=0); y = np.array(y_list)
scaler = StandardScaler(); Xs = scaler.fit_transform(X)
clf = LogisticRegression(max_iter=2000, random_state=42); clf.fit(Xs, y)

layer_module = model.model.layers[12]
vec_tensor = torch.from_numpy(steering_v).float()
eos_id = tokenizer.eos_token_id

for test_idx, sample in enumerate(test[:4]):
    label = sample.get("answerability")
    prompt = f"{sample.get('context','')}\n\nQuestion: {sample.get('question','')}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    gate_decided = False
    effective_alpha = 0.0
    probe_score = 0.5
    hook_call_count = [0]

    def make_hook():
        def hook(module, inputs, outputs):
            nonlocal gate_decided, effective_alpha, probe_score
            hook_call_count[0] += 1

            if isinstance(outputs, tuple):
                h_full = outputs[0]
            else:
                h_full = outputs

            if not gate_decided:
                h = h_full[0] if h_full.dim() == 3 else h_full
                pooled = h[-1, :].detach().cpu().float().numpy()
                X_s = scaler.transform(pooled.reshape(1, -1))
                proba = clf.predict_proba(X_s)[0, 1]
                probe_score = float(proba)
                gate_decided = True
                if probe_score < 0.5:
                    effective_alpha = -1.0
                    print(f'  [{test_idx}] PREFILL: gate=1, alpha=-1.0 (probe={proba:.4f})')
                else:
                    effective_alpha = 0.0
                    print(f'  [{test_idx}] PREFILL: gate=0, pass through (probe={proba:.4f})')

            if abs(effective_alpha) > 0.001:
                v = (-1.0) * vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
                h_modified = h_full + v
                if isinstance(outputs, tuple):
                    return (h_modified,) + outputs[1:]
                else:
                    return h_modified
            else:
                return None

        return hook

    handle = layer_module.register_forward_hook(make_hook())
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    pkv = outputs.past_key_values
    generated_ids = []
    current_input = input_ids
    for _step in range(48):
        with torch.no_grad():
            if pkv is not None:
                current_input = current_input[:, -1:]
            out = model(input_ids=current_input, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tid = next_token.item()
            generated_ids.append(tid)
            if tid == eos_id:
                break
            current_input = next_token

    handle.remove()
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(f'  [{test_idx}] label={label}, hook_calls={hook_call_count[0]}, tokens={len(generated_ids)}, output={answer[:100]}')

print('\nDone!')