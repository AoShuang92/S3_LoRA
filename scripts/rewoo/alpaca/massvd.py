import torch

import torch.nn.utils.prune as prune

def massvd(pmodel, proj_modules=None, adapter_name='default', top_k=5):
    """

    Args:
        pmodel: The model object with LoRA adapters.
        proj_modules: List of projection module names (e.g., ['q_proj', 'k_proj', 'v_proj', 'o_proj']).
        adapter_name: Name of the LoRA adapter to process.
        top_k: Number of top singular vectors to consider when searching best (u, v) pair.

    Returns:
        specs: List of top singular values (after magnitude recovery).
        all_singular_values: List of all singular values per layer/module.
        vh_matrices: List of Vh matrices from final SVD.
    """
    if proj_modules is None:
        proj_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    specs, all_singular_values, vh_matrices = [], [], []

    for layer_idx, layer in enumerate(pmodel.base_model.model.model.layers):
        for proj_name in proj_modules:
            proj_module = getattr(layer.self_attn, proj_name, None)
            if proj_module is not None:
                if adapter_name in proj_module.lora_A and adapter_name in proj_module.lora_B:
                    lora_A = proj_module.lora_A[adapter_name].weight  # [r, in_dim]
                    lora_B = proj_module.lora_B[adapter_name].weight  # [out_dim, r]
                    vec = lora_B @ lora_A  # [out_dim, in_dim]

                    try:
                        # Step 1: Row and column normalization
                        row_norms = torch.norm(vec, dim=1, keepdim=True) + 1e-8
                        X_row = vec / row_norms
                        col_norms = torch.norm(vec, dim=0, keepdim=True) + 1e-8
                        X_col = vec / col_norms

                        # Step 2: SVD on row-normalized and column-normalized matrices
                        _, _, Vh_row = torch.linalg.svd(X_row, full_matrices=False)
                        U_col, _, _ = torch.linalg.svd(X_col, full_matrices=False)

                        # Step 3: Find best (u, v, d) minimizing L1 reconstruction error
                        best_score = float('inf')
                        best_u, best_v, best_d = None, None, None
                        for u in U_col[:top_k]:
                            for v in Vh_row[:top_k]:
                                d = torch.median((vec @ v) * u)
                                recon = d * torch.ger(u, v)
                                score = torch.norm(vec - recon, p=1)
                                if score < best_score:
                                    best_score = score
                                    best_u, best_v, best_d = u, v, d

                        # Step 4: Magnitude preservation: rescale using original row norms
                        recon_normalized = best_d * torch.ger(best_u, best_v)
                        recon_rescaled = row_norms * recon_normalized  # magnitude-preserved matrix

                        # Step 5: Final SVD on rescaled matrix
                        _, S, Vh_final = torch.linalg.svd(recon_rescaled, full_matrices=False)

                        # Step 6: Store results
                        specs.append(S[0].item())  # top singular value
                        all_singular_values.append(S.cpu().numpy())
                        vh_matrices.append(Vh_final)

                        print(f"✅ Layer {layer_idx} | {proj_name} | SpSVD successful | shape: {vec.shape}")

                    except Exception as e:
                        specs.append(float('nan'))
                        print(f"⚠️ SpSVD failed at layer {layer_idx} {proj_name}: {e}")

    return specs, all_singular_values, vh_matrices


import numpy as np

def sharp_index_score(all_singular_values, top_k=5, verbose=True):
    """
    Compute SharpIndex for each layer from singular values and identify top-k sharpest layers.

    Args:
        all_singular_values: List of singular value arrays, one per layer/module.
        top_k: Number of top sharp (outlier) layers to return.
        verbose: If True, prints detected outlier layers.

    Returns:
        sharp_indices: np.array of SharpIndex scores.
        outlier_indices: List of indices corresponding to top-k sharp layers.
    """
    sharp_indices = []

    for svals in all_singular_values:
        if len(svals) == 0 or np.sum(svals) == 0:
            sharp_indices.append(np.nan)
        else:
            sharp_index = svals[0] / (np.sum(svals) + 1e-6)
            sharp_indices.append(sharp_index)

    sharp_indices = np.array(sharp_indices)
    sorted_indices = np.argsort(-sharp_indices)  # Descending sort
    outlier_indices = sorted_indices[:top_k]

    if verbose:
        print("\n🚨 Detected Outlier Layers by SharpIndex:")
        for idx in outlier_indices:
            print(f"⚠️  Layer Index {idx} | SharpIndex = {sharp_indices[idx]:.4f}")

    return sharp_indices, outlier_indices

def zero_out_layers(pmodel, outlier_indices, proj_modules=None, adapter_name='default'):
    """
    Zero out lora_B weights in specified outlier layers/modules by global index.

    Args:
        pmodel: The model object containing LoRA adapters.
        outlier_indices: List or array of global indices to zero out (e.g., from SharpIndex).
        proj_modules: List of projection modules to process (default: ['q_proj', 'k_proj', 'v_proj', 'o_proj']).
        adapter_name: Name of the LoRA adapter to modify.
    """
    if proj_modules is None:
        proj_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    global_idx = 0  # flat index across all layers and proj_modules

    for layer_idx, layer in enumerate(pmodel.base_model.model.model.layers):
        for proj_name in proj_modules:
            if global_idx in outlier_indices:
                proj_module = getattr(layer.self_attn, proj_name, None)
                if proj_module is not None:
                    if adapter_name in proj_module.lora_B:
                        lora_B = proj_module.lora_B[adapter_name].weight
                        lora_B.data.zero_()
                        print(f"🧼 Zeroed lora_B in layer {layer_idx} | {proj_name} | global idx {global_idx}")
            global_idx += 1
    return pmodel

    
# Function to count zero parameters in each layer
def count_zero_params(model):
    zero_params = {}
    for name, param in model.named_parameters():
        num_zeros = torch.sum(param == 0).item()
        total_params = param.numel()
        sparsity = num_zeros / total_params * 100
        zero_params[name] = (num_zeros, total_params, sparsity)
    return zero_params

# Function to prune layers with high sparsity and return pruned model
def prune_zero_params(model, threshold=50):
    pruned_model = model  # Work on a copy of the model

    for name, module in pruned_model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            num_zeros = torch.sum(module.weight == 0).item()
            total_params = module.weight.numel()
            sparsity = (num_zeros / total_params) * 100

            if sparsity > threshold:
                # print(f"Pruning {name} with sparsity {sparsity:.2f}%")
                prune.l1_unstructured(module, name="weight", amount=1.0)  # Fully prune
                prune.remove(module, "weight")  # Remove redundant params

    return pruned_model  # Return the pruned model
