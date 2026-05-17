"""Quick test of single-pass hook-based gate on one sample."""
import os, sys, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model_loader import load_model_and_tokenizer, get_model_layer_count
from src.activation_collector import load_activations
from src.steering import get_all_vectors
from src.data_builder import load_jsonl

model, tokenizer = load_model_and_tokenizer('Qwen/Qwen2.5-0.5B-Instruct', 'cpu', 'float32')
print(f'Layers: {get_model_layer_count(model)}')

train = load_jsonl('data_m3/train_s0.jsonl')
test = load_jsonl('data_m3/test_s0.jsonl')
print(f'Train: {len(train)}, Test: {len(test)}')

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
print(f'Probe train_acc={accuracy_score(y, clf.predict(Xs)):.4f}')

sample = test[0]
print(f"Test sample[0]: answerability={sample.get('answerability')}")
prompt = f"{sample.get('context','')}\n\nQuestion: {sample.get('question','')}\nAnswer:"
inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)

layer_module = model.model.layers[12]
vec_tensor = torch.from_numpy(steering_v).float()

def gate_hook(module, hook_inputs, hook_outputs):
    if isinstance(hook_outputs, tuple):
        h_full = hook_outputs[0]
    else:
        h_full = hook_outputs
    h = h_full[0] if h_full.dim() == 3 else h_full
    
    print(f'  Hook: outputs type={type(hook_outputs)}, h_full.shape={h_full.shape}, h.shape={h.shape}')
    
    pooled = h[-1, :].detach().cpu().float().numpy()
    X_sample = scaler.transform(pooled.reshape(1, -1))
    proba = clf.predict_proba(X_sample)[0, 1]
    print(f'  Hook: probe_score={proba:.4f}')
    
    if proba >= 0.5:
        print(f'  Hook: gate=0, pass through (return None)')
        return None
    else:
        print(f'  Hook: gate=1, apply steering')
        v = (-1.0) * vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
        h_modified = h_full + v
        if isinstance(hook_outputs, tuple):
            return (h_modified,) + hook_outputs[1:]
        else:
            return h_modified

handle = layer_module.register_forward_hook(gate_hook)
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
handle.remove()
print(f'Done! Output hidden_states shape: {outputs.hidden_states[-1].shape}')