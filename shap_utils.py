import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau, pearsonr, entropy as scipy_entropy
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import warnings
from scipy.stats import entropy

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def safe_forward(model, x):
    x = x.clone().requires_grad_(True)
    original_states = {}
    for name, module in model.named_modules():
        if hasattr(module, 'inplace'):
            original_states[name] = module.inplace
            module.inplace = False
    try:
        with torch.enable_grad():
            if hasattr(model, 'gnn_transform'):
                x = model.gnn_transform(x)
            features = model.featurizer(x)
            bottleneck = model.bottleneck(features)
            output = model.classifier(bottleneck)
            return output
    finally:
        for name, module in model.named_modules():
            if name in original_states:
                module.inplace = original_states[name]

class PredictWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return safe_forward(self.model, x)

def get_background_batch(loader, size=64):
    background = []
    for batch in loader:
        background.append(batch[0])
        if len(background) >= size:
            break
    return torch.cat(background, dim=0)[:size]

def safe_compute_shap_values(model, background, inputs, nsamples=200):
    wrapped_model = PredictWrapper(model)
    # Try DeepExplainer, fallback to GradientExplainer if necessary
    try:
        explainer = shap.DeepExplainer(wrapped_model, background)
        shap_values = explainer.shap_values(inputs, check_additivity=False)
    except Exception:
        explainer = shap.GradientExplainer(wrapped_model, background)
        shap_values = explainer.shap_values(inputs)
    # Convert to SHAP Explanation object for better handling
    return shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=to_numpy(inputs)
    )

def _get_shap_array(shap_values):
    if isinstance(shap_values, list):
        return shap_values[0].values if hasattr(shap_values[0], "values") else shap_values[0]
    elif hasattr(shap_values, 'values'):
        return shap_values.values
    return shap_values

def plot_summary(shap_values, features, output_path, max_display=20):
    plt.figure(figsize=(10, 6))
    shap_array = _get_shap_array(shap_values)
    flat_features = features.reshape(features.shape[0], -1)
    flat_shap_values = shap_array.reshape(shap_array.shape[0], -1)
    if flat_shap_values.shape != flat_features.shape:
        min_samples = min(flat_shap_values.shape[0], flat_features.shape[0])
        min_features = min(flat_shap_values.shape[1], flat_features.shape[1])
        flat_shap_values = flat_shap_values[:min_samples, :min_features]
        flat_features = flat_features[:min_samples, :min_features]
    feature_names = [f"F{i}" for i in range(flat_features.shape[1])]
    shap.summary_plot(
        flat_shap_values,
        flat_features,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
        rng=42
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved summary plot: {output_path}")

def overlay_signal_with_shap(signal, shap_vals, output_path):
    signal = to_numpy(signal)
    shap_vals = _get_shap_array(shap_vals)
    shap_vals = to_numpy(shap_vals)
    if signal.ndim > 1:
        signal = signal.squeeze()
    if shap_vals.ndim > 1:
        shap_vals = shap_vals.squeeze()
    signal_flat = signal.reshape(-1)
    shap_vals_flat = np.abs(shap_vals).reshape(-1)
    min_len = min(len(signal_flat), len(shap_vals_flat))
    signal_flat = signal_flat[:min_len]
    shap_vals_flat = shap_vals_flat[:min_len]
    plt.figure(figsize=(12, 6))
    plt.plot(signal_flat, label="Signal", color="steelblue", alpha=0.7, linewidth=1.5)
    plt.fill_between(
        np.arange(min_len),
        0,
        shap_vals_flat,
        color="red",
        alpha=0.3,
        label="|SHAP|"
    )
    plt.title("Signal with SHAP Overlay")
    plt.xlabel("Flattened Feature Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved signal overlay: {output_path}")

def plot_shap_heatmap(shap_values, output_path):
    shap_vals = _get_shap_array(shap_values)
    abs_vals = np.abs(to_numpy(shap_vals))
    while abs_vals.ndim > 2:
        abs_vals = abs_vals.mean(axis=tuple(range(abs_vals.ndim - 2)))
    if abs_vals.ndim != 2:
        raise ValueError(f"Could not reduce SHAP values to 2D array. Final shape: {abs_vals.shape}")
    aggregated = abs_vals.T
    plt.figure(figsize=(12, 8))
    plt.imshow(aggregated,
               aspect='auto',
               cmap='viridis',
               interpolation='nearest')
    plt.colorbar(label='|SHAP Value|')
    plt.xlabel("Time Steps")
    plt.ylabel("Channels")
    plt.title("SHAP Value Heatmap")
    if aggregated.shape[0] <= 8:
        plt.yticks(range(aggregated.shape[0]), [f"CH{i+1}" for i in range(aggregated.shape[0])])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved SHAP heatmap: {output_path}")

def evaluate_shap_impact(model, inputs, shap_values, top_k=0.2):
    model.eval()
    with torch.no_grad():
        base_preds = model.predict(inputs)
        base_preds = torch.softmax(base_preds, dim=1)
    base_preds_np = to_numpy(base_preds)
    inputs_np = to_numpy(inputs)
    shap_vals_np = to_numpy(_get_shap_array(shap_values))
    if inputs_np.ndim == 3:
        inputs_np = inputs_np[:, :, np.newaxis, :]
        shap_vals_np = shap_vals_np[:, :, np.newaxis, :]
    batch_size, n_channels, n_spatial, n_timesteps = inputs_np.shape
    masked_inputs = inputs_np.copy()
    for i in range(batch_size):
        importance = np.abs(shap_vals_np[i]).mean(axis=(0, 1))
        if len(importance) > n_timesteps:
            importance = importance[:n_timesteps]
        k = int(n_timesteps * top_k)
        top_indices = np.argsort(importance)[-k:]
        top_indices = top_indices[top_indices < n_timesteps]
        masked_inputs[i, :, :, top_indices] = 0
    masked_tensor = torch.tensor(masked_inputs, dtype=inputs.dtype).to(inputs.device)
    with torch.no_grad():
        masked_preds = model.predict(masked_tensor)
        masked_preds = torch.softmax(masked_preds, dim=1)
    base_classes = base_preds.argmax(dim=1)
    masked_classes = masked_preds.argmax(dim=1)
    acc_drop = 100 * (1 - (base_classes == masked_classes).float().mean().item())
    return to_numpy(base_preds), to_numpy(masked_preds), acc_drop

def compute_flip_rate(base_preds, masked_preds):
    base_classes = np.argmax(base_preds, axis=1)
    masked_classes = np.argmax(masked_preds, axis=1)
    flip_rate = np.mean(base_classes != masked_classes)
    return flip_rate

def compute_confidence_change(base_preds, masked_preds):
    base_conf = np.max(base_preds, axis=1)
    masked_conf = np.max(masked_preds, axis=1)
    conf_change = np.mean(base_conf - masked_conf)
    return conf_change

def compute_aopc(model, inputs, shap_values, steps=10):
    model.eval()
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(2)
    inputs_np = to_numpy(inputs)
    batch_size, n_channels, n_spatial, n_timesteps = inputs_np.shape
    device = inputs.device
    shap_vals_np = to_numpy(_get_shap_array(shap_values))
    if shap_vals_np.ndim == 3:
        shap_vals_np = shap_vals_np[:, :, np.newaxis, :]
    with torch.no_grad():
        base_preds = model.predict(inputs)
        base_conf = torch.softmax(base_preds, dim=1).max(dim=1).values.cpu().numpy()
    aopc_scores = []
    for i in range(batch_size):
        importance = np.abs(shap_vals_np[i]).mean(axis=(0, 1))
        if len(importance) > n_timesteps:
            importance = importance[:n_timesteps]
        sorted_indices = np.argsort(importance)[::-1].copy()
        mask_indices_tensor = torch.from_numpy(sorted_indices).to(device)
        current_input = inputs[i].clone().detach()
        original_conf = base_conf[i]
        confidences = [original_conf]
        for step in range(1, steps + 1):
            k = int(n_timesteps * step / steps)
            mask_indices = mask_indices_tensor[:k]
            mask_indices = mask_indices[mask_indices < n_timesteps]
            modified_input = current_input.clone()
            if modified_input.dim() == 3:
                modified_input[:, :, mask_indices] = 0
            else:
                modified_input[..., mask_indices] = 0
            with torch.no_grad():
                pred = model.predict(modified_input.unsqueeze(0))
                conf = torch.softmax(pred, dim=1).max().item()
            confidences.append(conf)
        incremental_drops = []
        for j in range(1, len(confidences)):
            incremental_drop = confidences[j-1] - confidences[j]
            incremental_drops.append(incremental_drop)
        aopc = np.mean(incremental_drops) if incremental_drops else 0
        aopc_scores.append(aopc)
    return np.mean(aopc_scores)

def compute_shap_entropy(shap_values):
    abs_vals = np.abs(to_numpy(_get_shap_array(shap_values)))
    flat_vals = abs_vals.reshape(abs_vals.shape[0], -1)
    normalized = flat_vals / (flat_vals.sum(axis=1, keepdims=True) + 1e-10)
    ent = entropy(normalized, axis=1)
    return np.mean(ent)

def compute_feature_coherence(shap_values):
    vals = to_numpy(_get_shap_array(shap_values))
    channel_corrs = []
    for i in range(vals.shape[0]):
        chan_vals = vals[i].squeeze()
        if chan_vals.ndim > 2:
            chan_vals = chan_vals.reshape(chan_vals.shape[0], -1)
        if chan_vals.shape[0] == 1:
            channel_corrs.append(1.0)
            continue
        corr_matrix = np.corrcoef(chan_vals)
        np.fill_diagonal(corr_matrix, 0)
        channel_corrs.append(np.mean(corr_matrix))
    return np.mean(channel_corrs) if channel_corrs else 0

def compute_pca_alignment(shap_values):
    vals = to_numpy(_get_shap_array(shap_values))
    flat_vals = vals.reshape(vals.shape[0], -1)
    pca = PCA(n_components=2)
    pca.fit(np.abs(flat_vals))
    return pca.explained_variance_ratio_.sum()

def evaluate_advanced_shap_metrics(shap_values, inputs):
    shap_vals = to_numpy(_get_shap_array(shap_values))
    inputs_np = to_numpy(inputs)
    flat_inputs = inputs_np.reshape(-1)
    flat_shap = np.abs(shap_vals).reshape(-1)
    if len(flat_inputs) > 1000:
        idx = np.random.choice(len(flat_inputs), 1000, replace=False)
        flat_inputs = flat_inputs[idx]
        flat_shap = flat_shap[idx]
    input_min = np.min(flat_inputs)
    input_max = np.max(flat_inputs)
    input_bins = np.digitize(flat_inputs, bins=np.linspace(input_min, input_max, 10))
    shap_min = 0
    shap_max = np.max(flat_shap) + 1e-10
    shap_bins = np.digitize(flat_shap, bins=np.linspace(shap_min, shap_max, 10))
    metrics = {
        'shap_entropy': compute_shap_entropy(shap_values),
        'feature_coherence': compute_feature_coherence(shap_values),
        'channel_variance': np.var(shap_vals, axis=(0, 2, 3)).mean() if shap_vals.ndim == 4 else 0,
        'temporal_entropy': entropy(np.abs(shap_vals).mean(axis=(0, 1, 2)).ravel()) if shap_vals.ndim == 4 else 0,
        'mutual_info': mutual_info_score(input_bins, shap_bins),
        'pca_alignment': compute_pca_alignment(shap_values)
    }
    return {k: float(v) for k, v in metrics.items()}

def plot_emg_shap_4d(inputs, shap_values, output_path):
    if not output_path.endswith('.html'):
        output_path = os.path.splitext(output_path)[0] + ".html"
    inputs = to_numpy(inputs)
    shap_vals = to_numpy(_get_shap_array(shap_values))
    sample_idx = 0
    inputs = inputs[sample_idx]
    shap_vals = shap_vals[sample_idx]
    shap_vals = np.squeeze(shap_vals)
    n_timesteps = inputs.shape[-1]
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)
    elif shap_vals.ndim > 1:
        shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)
        shap_vals = shap_vals[:, :n_timesteps]
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, n_timesteps)
    elif shap_vals.ndim > 2:
        shap_vals = shap_vals.reshape(-1, n_timesteps)
    n_channels = shap_vals.shape[0]
    time_steps = np.arange(n_timesteps)
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    for ch in range(n_channels):
        shap_mag = np.abs(shap_vals[ch])
        if len(shap_mag) != len(time_steps):
            min_len = min(len(shap_mag), len(time_steps))
            shap_mag = shap_mag[:min_len]
            ch_time_steps = time_steps[:min_len]
        else:
            ch_time_steps = time_steps
        fig.add_trace(go.Scatter3d(
            x=ch_time_steps,
            y=np.full_like(ch_time_steps, ch),
            z=shap_mag,
            mode='lines',
            name=f'Channel {ch+1}',
            line=dict(width=4)
        ))
    fig.update_layout(
        title='4D SHAP Value Distribution (Sample 0)',
        scene=dict(
            xaxis_title='Time Steps',
            yaxis_title='Channels',
            zaxis_title='|SHAP Value|'
        ),
        height=800,
        width=1000
    )
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"✅ Saved interactive 4D SHAP plot: {output_path}")

def plot_4d_shap_surface(shap_values, output_path):
    if not output_path.endswith('.html'):
        output_path = os.path.splitext(output_path)[0] + ".html"
    shap_vals = to_numpy(_get_shap_array(shap_values))
    shap_vals = np.squeeze(shap_vals)
    if shap_vals.ndim == 4:
        shap_vals = shap_vals.mean(axis=2)
    if shap_vals.ndim == 3:
        aggregated = np.abs(shap_vals).mean(axis=0)
    elif shap_vals.ndim == 2:
        aggregated = np.abs(shap_vals)
    else:
        raise ValueError(f"Unsupported SHAP dimension: {shap_vals.ndim}")
    if aggregated.shape[0] > aggregated.shape[1]:
        aggregated = aggregated.T
    channels = np.arange(aggregated.shape[0])
    time_steps = np.arange(aggregated.shape[1])
    X, Y = np.meshgrid(time_steps, channels)
    fig = go.Figure(data=[
        go.Surface(
            z=aggregated,
            x=X,
            y=Y,
            colorscale='Viridis',
            opacity=0.9,
            contours={
                "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
            }
        )
    ])
    fig.update_layout(
        title='SHAP Value Surface (Avg Across Samples)',
        scene=dict(
            xaxis_title='Time Steps',
            yaxis_title='Channels',
            zaxis_title='|SHAP Value|',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800,
        width=1000
    )
    fig.update_layout(coloraxis_colorbar=dict(
        title="|SHAP|",
        thickness=15,
        len=0.5
    ))
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"✅ Saved interactive SHAP surface plot: {output_path}")

def compute_kendall_tau(shap1, shap2):
    flat1 = np.abs(shap1).flatten()
    flat2 = np.abs(shap2).flatten()
    return kendalltau(flat1, flat2)[0]

def cosine_similarity_shap(shap1, shap2):
    flat1 = np.abs(shap1).flatten()
    flat2 = np.abs(shap2).flatten()
    return 1 - cosine(flat1, flat2)

def log_shap_values(shap_array):
    abs_shap = np.abs(shap_array)
    return np.log(abs_shap + 1e-12)

def compute_jaccard_topk(shap1, shap2, k=10):
    flat1 = np.abs(shap1).flatten()
    flat2 = np.abs(shap2).flatten()
    top1 = set(np.argsort(-flat1)[:k])
    top2 = set(np.argsort(-flat2)[:k])
    intersection = len(top1.intersection(top2))
    union = len(top1.union(top2))
    return intersection / union if union > 0 else 0

def save_shap_numpy(shap_values, save_path="shap_values.npy"):
    shap_array = _get_shap_array(shap_values)
    np.save(save_path, shap_array)
    print(f"✅ Saved SHAP values to: {save_path}")
