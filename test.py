
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.metrics import (
    f1_score, average_precision_score, roc_curve, 
    precision_recall_curve, roc_auc_score, auc,
    precision_score, recall_score
)
import pickle
import datetime
aff_path = "/root/autodl-tmp/MORF/Transformer/"
if aff_path not in sys.path:
    sys.path.append(aff_path)
batch_size = 32 
num_epochs = 100 
patience = 5    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 1 
max_length = 11
PROT_FEAT_DIM = 1024 
ESM_FEAT_DIM = 2560   
TOTAL_FEAT_DIM = PROT_FEAT_DIM + ESM_FEAT_DIM  
ENABLE_BAYES_CORRECTION = True  
ENABLE_SMOOTHING = True       
SMOOTH_WINDOW_SIZE = 5         
BAYES_PRIOR_DEFAULT = 0.5  

def residue_level_bayes_evidence(residue_probs, prior=0.5):

    if not ENABLE_BAYES_CORRECTION:
        return residue_probs  
    
    if isinstance(residue_probs, torch.Tensor):
        posterior = residue_probs.cpu().numpy()
    else:
        posterior = residue_probs
    numerator = posterior * (1 - prior)
    denominator = posterior + prior - 2 * posterior * prior
    denominator = np.where(denominator == 0, 1e-8, denominator)
    denominator = np.clip(denominator, 1e-8, 1e8)
    corrected = numerator / denominator
    corrected = np.clip(corrected, 0.0, 1.0)
    corrected_tensor = torch.tensor(corrected, dtype=torch.float32, device=residue_probs.device)
    return corrected_tensor
def smooth_scores_per_sequence(scores, window_metas, window_size=SMOOTH_WINDOW_SIZE):

    if not ENABLE_SMOOTHING:
        return scores 
    
    if isinstance(scores, torch.Tensor):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = scores

    seq_ids = window_metas[:, 0]  
    unique_seq_ids = np.unique(seq_ids)

    smoothed_scores_np = np.copy(scores_np)

    smooth_window_size = 5

    for seq_id in unique_seq_ids:
        seq_windows = window_metas[seq_ids == seq_id]
        seq_start = min(win[3] for win in seq_windows)  
        seq_end = max(win[4] for win in seq_windows)   

        seq_scores = scores_np[seq_start:seq_end, :].squeeze(axis=1)  # (N,)
        seq_len = len(seq_scores)
        
        if seq_len < smooth_window_size:
            continue

        s1 = np.concatenate(([seq_scores[0]], seq_scores))
        s0 = np.concatenate(([s1[0]], s1))
        s3 = np.concatenate((seq_scores, [seq_scores[-1]]))
        s4 = np.concatenate((s3, [s3[-1]]))
        
        seq_smoothed = (s0[:-2] + s1[:-1] + seq_scores + s3[1:] + s4[2:]) / 5.0

        if len(seq_smoothed) != seq_len:
            seq_smoothed = seq_scores  

        smoothed_scores_np[seq_start:seq_end, :] = seq_smoothed.reshape(-1, 1)

    smoothed_tensor = torch.tensor(smoothed_scores_np, dtype=torch.float32, device=scores.device)
    return smoothed_tensor

class GatedFeatureFusion(nn.Module):

    def __init__(self, prot_dim=PROT_FEAT_DIM, esm_dim=ESM_FEAT_DIM):
        super(GatedFeatureFusion, self).__init__()
        self.prot_dim = prot_dim
        self.esm_dim = esm_dim
        self.total_dim = prot_dim + esm_dim  

        self.gate_linear = nn.Linear(self.total_dim, 2)

        nn.init.xavier_uniform_(self.gate_linear.weight)
        if self.gate_linear.bias is not None:
            nn.init.constant_(self.gate_linear.bias, 0.0)
    
    def forward(self, x):
        assert x.shape[-1] == self.total_dim, f"输入特征维度{x.shape[-1]}与门控模块配置维度{self.total_dim}不匹配"
        prot_feat = x[..., :self.prot_dim] 
        esm_feat = x[..., self.prot_dim:] 
        
        # 计算门控权重
        gate_scores = self.gate_linear(x)  
        gate_weights = F.softmax(gate_scores, dim=-1) 
        g1 = gate_weights[..., 0:1]  
        g2 = gate_weights[..., 1:2]  

        weighted_prot = prot_feat * g1
        weighted_esm = esm_feat * g2

        fused_feat = torch.cat([weighted_prot, weighted_esm], dim=-1)
        
        return fused_feat

class ProteinDataset(Dataset):
    def __init__(self, window_feats, residue_labels_list, meta_indices):
        self.window_feats = window_feats 
        self.residue_labels_list = residue_labels_list  
        self.meta_indices = meta_indices 
        self.segment_true_labels = self._precompute_segment_labels()  

    def _precompute_segment_labels(self):
        segment_labels = []
        for residue_labels in self.residue_labels_list:
            if isinstance(residue_labels, torch.Tensor):
                residue_labels = residue_labels.numpy()
            elif not isinstance(residue_labels, np.ndarray):
                residue_labels = np.array(residue_labels)

            valid_mask = (residue_labels != -1).squeeze(-1)
            valid_residue_labels = residue_labels[valid_mask]

            if valid_residue_labels.size == 0:
                segment_label_discrete = torch.tensor([0.0], dtype=torch.float32)
            else:
                segment_label_mean = valid_residue_labels.mean(axis=0)  # (1,)
                segment_label_discrete = (segment_label_mean > 0.5).astype(np.float32)
                segment_label_discrete = torch.tensor(segment_label_discrete, dtype=torch.float32)
            
            segment_labels.append(segment_label_discrete)
        return segment_labels

    def __len__(self):
        return len(self.window_feats)

    def __getitem__(self, idx):
        return (
            self.window_feats[idx],  
            self.segment_true_labels[idx],  
            torch.tensor(self.meta_indices[idx], dtype=torch.long)
        )

def calculate_global_averaged_probs(predictions, metas, total_residues, num_classes=1):
    prob_sum = torch.zeros((total_residues, num_classes), dtype=torch.float32, device=device)
    prob_count = torch.zeros(total_residues, dtype=torch.int32, device=device)

    for pred, meta in zip(predictions, metas):
        _, _, _, global_start, global_end = meta
        actual_len = global_end - global_start
        pred_valid = pred[:actual_len, :]  # (actual_len, 1)

        prob_sum[global_start:global_end, :] += pred_valid
        prob_count[global_start:global_end] += 1

    global_averaged_probs = prob_sum / prob_count.unsqueeze(1).clamp(min=1)
    return global_averaged_probs

def aggregate_residue_labels_to_segment(residue_labels, aggregation='mean'):
    valid_mask = (residue_labels != -1).squeeze(-1)
    valid_residue_labels = residue_labels[valid_mask]
    
    if valid_residue_labels.size == 0:
        return torch.tensor([0.0], dtype=torch.float32)
    
    if aggregation == 'mean':
        segment_label = valid_residue_labels.mean(dim=0)  # (1,)
    else:
        raise ValueError(f"仅支持'mean'聚合规则，当前输入：{aggregation}")
    return segment_label

def split_by_windows(np_array, window_metas, max_window_len):
    tensor_list = []
    start_idx = 0
    for meta in window_metas:
        _, _, _, global_start, global_end = meta
        window_actual_len = global_end - global_start
        window_fragment = np_array[start_idx:start_idx + window_actual_len]
        pad_len = max_window_len - window_actual_len
        if pad_len > 0:
            window_fragment = np.pad(window_fragment, ((0, pad_len), (0, 0)), mode='constant')
        tensor_list.append(torch.tensor(window_fragment, dtype=torch.float32))
        start_idx += window_actual_len
    return tensor_list, list(range(len(window_metas)))

def convert_to_list_of_tensors(data: np.ndarray):
    return [torch.tensor(row, dtype=torch.float32).unsqueeze(0) for row in data]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None  

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape  
        if self.pe is None or self.pe.size(1) < seq_len:
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))

            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            if self.d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(0)  

        return x + self.pe[:, :seq_len, :]  

class CAFModule(nn.Module):
    def __init__(self, dim=256, dropout=0.3):
        super(CAFModule, self).__init__()
        self.dim = dim
       
        self.conv_fusion = nn.Conv1d(
            in_channels=2 * dim,  
            out_channels=dim,  
            kernel_size=1,     
            stride=1,
            padding=0,
            bias=False      
        )
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalized_shape=dim)

    def forward(self, z_current, z_history):
        z_current_perm = z_current.permute(0, 2, 1)  
        z_history_perm = z_history.permute(0, 2, 1)  

        z_concat = torch.stack([z_current_perm, z_history_perm], dim=1)  
        b, c, d, s = z_concat.shape  # b=batch, c=2, d=dim, s=seq_len

        z_concat = z_concat.transpose(1, 2).reshape(b, -1, s)  

        z_fused = self.conv_fusion(z_concat)  

        z_fused = z_fused.permute(0, 2, 1)  
        z_fused = self.gelu(z_fused)
        z_fused = self.dropout(z_fused)
        z_fused = self.norm(z_fused)

        return z_fused

class MultiScaleHybridAttention(nn.Module):
    def __init__(self, embed_size, heads, 
                 local_window_sizes=[3, 6],  
                 global_window_size=11):     
        super(MultiScaleHybridAttention, self).__init__()
        assert embed_size % heads == 0, "Embed size needs to be divisible by heads"
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.scale = self.head_dim ** -0.5  

        self.local_qkv = nn.ModuleList([
            nn.Linear(embed_size, embed_size * 3, bias=False) 
            for _ in local_window_sizes
        ])
        self.local_window_sizes = local_window_sizes

        self.global_qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.global_window_size = global_window_size

        self.fc_out = nn.Linear(embed_size, embed_size)

        self.alpha_local = nn.Parameter(torch.tensor(0.5))
        self.alpha_global = nn.Parameter(torch.tensor(0.5))

        max_window = max(local_window_sizes)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_window - 1), heads)  
        )
        
    def _get_relative_position_bias(self, seq_len, window_size, device):
        coords = torch.arange(seq_len, device=device)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords_clipped = torch.clamp(relative_coords, 
                                              -window_size + 1, window_size - 1)
        relative_coords_clipped = relative_coords_clipped + window_size - 1
        relative_position_bias = self.relative_position_bias_table[relative_coords_clipped.view(-1)].view(
            seq_len, seq_len, -1
        ).permute(2, 0, 1).unsqueeze(0)  # [1, heads, seq_len, seq_len]
        return relative_position_bias
    
    def _compute_local_attention(self, queries, keys, seq_len, window_size, device):
        N = queries.shape[0]

        attention_mask = torch.zeros(N, self.heads, seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            attention_mask[:, :, i, start:end] = 1
   
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) * self.scale

        relative_bias = self._get_relative_position_bias(seq_len, window_size, device)
        energy = energy + relative_bias

        energy = energy.masked_fill(attention_mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)
        
        return attention
    
    def _compute_global_attention(self, queries, keys, seq_len, device):
        N = queries.shape[0]

        attention_mask = torch.zeros(N, self.heads, seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.global_window_size // 2)
            end = min(seq_len, i + self.global_window_size // 2 + 1)
            attention_mask[:, :, i, start:end] = 1

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) * self.scale
        energy = energy.masked_fill(attention_mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)
        
        return attention
    
    def forward(self, x):
        N, seq_len, _ = x.shape  
        device = x.device
 
        x_multi_scale = x
        
        local_outs = []
        for idx, local_qkv in enumerate(self.local_qkv):
            window_size = self.local_window_sizes[idx]

            local_qkv_out = local_qkv(x_multi_scale).reshape(
                N, seq_len, 3, self.heads, self.head_dim
            )
            local_q, local_k, local_v = (
                local_qkv_out[:, :, 0], local_qkv_out[:, :, 1], local_qkv_out[:, :, 2]
            )

            local_attention = self._compute_local_attention(
                local_q, local_k, seq_len, window_size, device
            )

            local_out = torch.einsum("nhqk,nkhd->nqhd", [local_attention, local_v])
            local_out = local_out.reshape(N, seq_len, self.embed_size)
            local_outs.append(local_out)

        local_out = sum(local_outs) / len(local_outs)

        global_qkv_out = self.global_qkv(x_multi_scale).reshape(
            N, seq_len, 3, self.heads, self.head_dim
        )
        global_q, global_k, global_v = (
            global_qkv_out[:, :, 0], global_qkv_out[:, :, 1], global_qkv_out[:, :, 2]
        )

        global_attention = self._compute_global_attention(
            global_q, global_k, seq_len, device
        )

        global_out = torch.einsum("nhqk,nkhd->nqhd", [global_attention, global_v])
        global_out = global_out.reshape(N, seq_len, self.embed_size)

        combined_out = self.alpha_local * local_out + self.alpha_global * global_out
        out = self.fc_out(combined_out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion,
                 local_window_sizes=[2,4,6], global_window_size=11):  
        super(TransformerBlock, self).__init__()
        self.attention = MultiScaleHybridAttention(
            embed_size=embed_size,
            heads=heads,
            local_window_sizes=local_window_sizes,
            global_window_size=global_window_size 
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, max_length, caf_dim=256,local_window_sizes=[2,4,6], global_window_size=11):  # 新增参数
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion,
                             local_window_sizes=local_window_sizes,
                             global_window_size=global_window_size) 
             for _ in range(num_layers)]
        )
        self.caf_modules = nn.ModuleList([
            CAFModule(dim=caf_dim, dropout=dropout) for _ in range(num_layers - 2)
        ])

    def forward(self, x):
        
        encoder_feats = []  
        
        for idx, layer in enumerate(self.layers):
            if idx < 2:
                x = layer(x)
                encoder_feats.append(x)
            else:
                x = layer(x)
                x = self.caf_modules[idx - 2](
                    z_current=x, 
                    z_history=encoder_feats[idx - 2]  
                )
                encoder_feats.append(x)
        
        return x

class ProteinTransformer(nn.Module):
    def __init__(self, embed_dim=3584, num_heads=8, num_layers=6, forward_expansion=4, 
                 dropout=0.1, max_length=max_length, num_classes=num_classes, 
                 hidden_dim=256, augment_eps=0.05):
        super(ProteinTransformer, self).__init__()
        self.gated_fusion = GatedFeatureFusion(
            prot_dim=PROT_FEAT_DIM,
            esm_dim=ESM_FEAT_DIM
        )

        self.input_block = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6),  
            nn.Linear(embed_dim, hidden_dim),   
            nn.LeakyReLU()
        )
        self.hidden_block = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim, eps=1e-6)
        )
        self.augment_eps = augment_eps
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        self.encoder = Encoder(
            hidden_dim, num_layers, num_heads, forward_expansion, 
            dropout, max_length, caf_dim=hidden_dim,
            local_window_sizes=[3,6],  
            global_window_size=11
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(hidden_dim, num_classes)  
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        if self.training and self.augment_eps > 0:
            src = src + self.augment_eps * torch.randn_like(src)

        fused_src = self.gated_fusion(src)

        src_input = self.input_block(fused_src) 
        src_hidden = self.pos_encoder(src_input)
        src_finish = self.hidden_block(src_hidden)
        output_encoder = self.encoder(src_finish)  

        output_permuted = output_encoder.permute(0, 2, 1)
        segment_feat = self.global_pool(output_permuted).squeeze(-1)
        segment_logits = self.fc_out(segment_feat)  

        return segment_logits

def calculate_sr(true_labels, pred_scores):
    valid_mask = (true_labels != -1).flatten()
    valid_true = true_labels[valid_mask].flatten()
    valid_pred = pred_scores[valid_mask].flatten()
    
    pos_mask = (valid_true == 1)
    neg_mask = (valid_true == 0)
    
    if not np.any(pos_mask) or not np.any(neg_mask):
        return 0.0
    
    pos_mean = np.mean(valid_pred[pos_mask])
    neg_mean = np.mean(valid_pred[neg_mask])
    
    return 1.0 if pos_mean > neg_mean else 0.0

def calculate_overall_sr(true_labels_np, fused_probs_np, test_window_metas):
    seq_ids = test_window_metas[:, 0]
    unique_seq_ids = np.unique(seq_ids)
    
    sr_count = 0
    total_seq = len(unique_seq_ids)
    
    for seq_id in unique_seq_ids:
        seq_windows = test_window_metas[seq_ids == seq_id]
        seq_start = min(win[3] for win in seq_windows)
        seq_end = max(win[4] for win in seq_windows)
        
        seq_true = true_labels_np[seq_start:seq_end]
        seq_pred = fused_probs_np[seq_start:seq_end]
        
        seq_sr = calculate_sr(seq_true, seq_pred)
        if seq_sr == 1.0:
            sr_count += 1
    
    overall_sr = sr_count / total_seq if total_seq > 0 else 0.0
    return overall_sr

def test_model(model, test_loader, test_window_metas, test_total_residues, num_classes=1, 
               test_residue_labels_list=None):
    model.eval()
    device = next(model.parameters()).device

    test_residue_prob_sum = torch.zeros((test_total_residues, num_classes), dtype=torch.float32, device=device)
    test_residue_count = torch.zeros(test_total_residues, dtype=torch.int32, device=device)
    test_residue_true = torch.zeros((test_total_residues, num_classes), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, (inputs, _, meta_indices) in enumerate(test_loader):
            inputs = inputs.to(device)

            segment_logits = model(inputs)
            segment_probs = torch.sigmoid(segment_logits)

            for idx in range(len(inputs)):
                meta_idx_in_dataset = meta_indices[idx].item()
                meta = test_window_metas[meta_idx_in_dataset]
                global_start, global_end = meta[3], meta[4]
                actual_window_len = global_end - global_start

                current_segment_prob = segment_probs[idx, :]
                residue_probs_for_window = current_segment_prob.unsqueeze(0).repeat(actual_window_len, 1)

                test_residue_prob_sum[global_start:global_end, :] += residue_probs_for_window
                test_residue_count[global_start:global_end] += 1

                true_residue_labels_for_window = test_residue_labels_list[meta_idx_in_dataset][:actual_window_len]
                if isinstance(true_residue_labels_for_window, np.ndarray):
                    true_residue_labels_for_window = torch.tensor(true_residue_labels_for_window, dtype=torch.float32, device=device)
                else:
                    true_residue_labels_for_window = true_residue_labels_for_window.to(device)
                test_residue_true[global_start:global_end, :] = true_residue_labels_for_window.clone().detach()

    global_averaged_probs = test_residue_prob_sum / test_residue_count.unsqueeze(1).clamp(min=1)

    label_valid_mask = (test_residue_true[:, 0] != -1)
    count_valid_mask = (test_residue_count > 0)
    val_mask = label_valid_mask & count_valid_mask
    
    if not torch.any(val_mask):
        print("警告：无有效数据用于评估！")
        return global_averaged_probs, torch.tensor([]), torch.tensor([]), test_residue_true, test_residue_count
    
    val_masked_probs = global_averaged_probs[val_mask]
    val_masked_labels = test_residue_true[val_mask]

    return global_averaged_probs, val_masked_labels, val_masked_probs, test_residue_true, test_residue_count

def save_roc_data(true_labels, pred_score, save_path):

    try:
        with open(save_path, "wb") as f:
            pickle.dump([true_labels, pred_score], f)
        print(f"ROC data saved to {save_path}")
    except Exception as e:
        print(f"Error saving ROC data: {e}")

def get_fpr_at_fixed_tpr_morf(true_labels, pred_scores, label_name='MoRFs', target_tprs=[0.2, 0.3, 0.4]):
    results = {}
    true_labels_np = true_labels.flatten() if isinstance(true_labels, torch.Tensor) else true_labels
    pred_scores_np = pred_scores.flatten() if isinstance(pred_scores, torch.Tensor) else pred_scores

    valid_mask = (true_labels_np != -1)
    true_labels_valid = true_labels_np[valid_mask]
    pred_scores_valid = pred_scores_np[valid_mask]

    binary_mask = (true_labels_valid == 0) | (true_labels_valid == 1)
    true_labels_binary = true_labels_valid[binary_mask]
    pred_scores_binary = pred_scores_valid[binary_mask]
    
    if len(true_labels_binary) == 0 or len(np.unique(true_labels_binary)) < 2:
        print("警告：无足够的二分类数据计算FPR")
        return {label_name: {}, 'Overall': {}}
    
    fpr, tpr, thresholds = roc_curve(true_labels_binary, pred_scores_binary)
    
    for target_tpr in target_tprs:
        idx = np.argmin(np.abs(tpr - target_tpr))
        achieved_tpr = tpr[idx]
        achieved_fpr = fpr[idx]
        threshold = thresholds[idx]
        
        results[target_tpr] = {
            "achieved_tpr": achieved_tpr,
            "fpr": achieved_fpr,
            "threshold": threshold
        }
    
    final_results = {label_name: results}
    final_results['Overall'] = results
    
    return final_results

def save_fpr_results_to_txt_morf(results, filename="fpr_results_morf.txt"):
    target_tprs = [0.2, 0.3, 0.4]
    label_name = 'MoRFs'
    
    with open(filename, 'w') as f:
        f.write("MoRFs Single Label FPR Results at Fixed TPR\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"校正开关：{ENABLE_BAYES_CORRECTION}，平滑开关：{ENABLE_SMOOTHING}，平滑窗口：{SMOOTH_WINDOW_SIZE}\n\n")
        
        for target_tpr in target_tprs:
            f.write(f"Target TPR: {target_tpr:.2f}\n")
            f.write("-" * 30 + "\n")
            
            if label_name in results and target_tpr in results[label_name]:
                metrics = results[label_name][target_tpr]
                f.write(f"{label_name}:\n")
                f.write(f"  Achieved TPR: {metrics['achieved_tpr']:.4f}\n")
                f.write(f"  FPR: {metrics['fpr']:.4f}\n")
                f.write(f"  Threshold: {metrics['threshold']:.4f}\n")
        
            if 'Overall' in results and target_tpr in results['Overall']:
                overall_metrics = results['Overall'][target_tpr]
                f.write(f"Overall:\n")
                f.write(f"  Achieved TPR: {overall_metrics['achieved_tpr']:.4f}\n")
                f.write(f"  FPR: {overall_metrics['fpr']:.4f}\n")
                f.write(f"  Threshold: {overall_metrics['threshold']:.4f}\n")
            
            f.write("\n")
    
    print(f"MoRFs FPR results saved to {filename}")

def save_comprehensive_metrics(true_labels_np, fused_probs_np, test_window_metas, save_path, model_weights=None):

    valid_mask = (true_labels_np != -1).flatten()
    true_labels_valid = true_labels_np.flatten()[valid_mask]
    fused_probs_valid = fused_probs_np.flatten()[valid_mask]

    pred_labels_valid = (fused_probs_valid >= 0.5).astype(int)

    try:
        precision = precision_score(true_labels_valid, pred_labels_valid, zero_division=0)
        recall = recall_score(true_labels_valid, pred_labels_valid, zero_division=0)
        f1 = f1_score(true_labels_valid, pred_labels_valid, zero_division=0)
        auc_score = roc_auc_score(true_labels_valid, fused_probs_valid)
    except ValueError as e:
        print(f"计算指标出错：{e}，使用默认值0")
        precision = recall = f1 = auc_score = 0.0

    sr = calculate_overall_sr(true_labels_np, fused_probs_np, test_window_metas)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(f"MoRFs 模型融合测试 - 综合评估指标（校正：{ENABLE_BAYES_CORRECTION}，平滑：{ENABLE_SMOOTHING}，窗口：{SMOOTH_WINDOW_SIZE}）\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if model_weights is not None:
            weight_desc = " + ".join([f"模型{i+1}({w})" for i, w in enumerate(model_weights)])
            f.write(f"融合权重: {weight_desc}\n\n")
        else:
            f.write(f"融合权重: 未指定（等权平均）\n\n")
        
        f.write(f"有效残基数: {len(true_labels_valid)}\n")
        f.write(f"MoRF残基数: {int(np.sum(true_labels_valid == 1))}\n\n")
        
        f.write(f"Precision (精确率): {precision:.4f}\n")
        f.write(f"Recall (召回率): {recall:.4f}\n")
        f.write(f"F1 Score (F1分数): {f1:.4f}\n")
        f.write(f"AUC (曲线下面积): {auc_score:.4f}\n")
        f.write(f"SR (Success Rate): {sr:.4f}\n")
    
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")
    print(f"   AUC: {auc_score:.4f}")
    print(f"   SR: {sr:.4f}")

def load_model_weights(model, model_path, model_name):
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_key = k[7:]
            else:
                new_key = k
            new_state_dict[new_key] = v
        gated_params_exist = any("gated_fusion" in k for k in new_state_dict.keys())
        if not gated_params_exist:
            print(f"⚠️  {model_name} 的state_dict中未找到门控模块参数，可能训练时未保存或结构不匹配")
        else:
            print(f"✅  {model_name} 找到门控模块参数，共{len([k for k in new_state_dict.keys() if 'gated_fusion' in k])}个")
        model.load_state_dict(new_state_dict, strict=False)

        gated_linear_weight = model.gated_fusion.gate_linear.weight
        if torch.allclose(gated_linear_weight, torch.nn.init.xavier_uniform_(torch.empty_like(gated_linear_weight))):
            print(f"⚠️  {model_name} 门控参数仍为默认初始化，可能加载失败")
        else:
            print(f"✅  {model_name} 门控参数已成功加载（非默认初始化）")
        
        print(f"✅ {model_name} state_dict加载成功")
        return model
    
    except Exception as e:
        print(f"❌ 加载{model_name} state_dict失败：{str(e)[:200]}")
        try:
            model = torch.load(model_path, map_location=device)
            print(f"✅ {model_name} 完整模型加载成功")
            return model
        except Exception as e2:
            print(f"❌ {model_name} 加载失败：{str(e2)[:200]}")
            exit()

# ======================== Traditional benchmark datasets ========================
if __name__ == "__main__":

    train_num_heads = 2   
    train_num_layers = 4  
    train_forward_expansion = 2  
    train_dropout = 0.6   
    train_hidden_dim = 256 
    train_augment_eps = 0.05 

    test_meta_path = "/root/autodl-tmp/MORF/Transformer/embedding/TEST464_metas.txt"
    
    def load_memmap_data(feat_path, label_path, feat_dim=2304, num_classes=3):
        feat_mmap = np.load(feat_path, mmap_mode='r')
        label_mmap = np.load(label_path, mmap_mode='r')
        return feat_mmap, label_mmap

    test_encodings_np, test_labels_np = load_memmap_data(
        "/root/autodl-tmp/MORF/Transformer/embedding/TEST464.npy",
        "/root/autodl-tmp/MORF/Transformer/embedding/TEST464_label.npy",
    )
    test_window_metas = np.loadtxt(test_meta_path, dtype=np.int32)

    test_total_residues = max(meta[4] for meta in test_window_metas) 
    test_max_window_len = max(meta[4] - meta[3] for meta in test_window_metas)

    test_encodings_windows, test_meta_indices = split_by_windows(test_encodings_np, test_window_metas, test_max_window_len)
    test_labels_windows, _ = split_by_windows(test_labels_np, test_window_metas, test_max_window_len)

    test_dataset = ProteinDataset(test_encodings_windows, test_labels_windows, test_meta_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("✅ Test loader loaded successfully")

    model_paths = [
        '/root/autodl-tmp/MORF/Transformer/model/olddata/inst2_auc_0.862_0.85.pth'
    ]
    priors_list = [BAYES_PRIOR_DEFAULT]  
    model_weights = [1.0] 

    true_labels_tensor = None
    global_probs_list = []
    test_residue_true = None
    test_residue_count = None
    
    for i, (model_path, prior, weight) in enumerate(zip(model_paths, priors_list, model_weights)):
        model = ProteinTransformer(
            embed_dim=TOTAL_FEAT_DIM,
            num_heads=train_num_heads,
            num_layers=train_num_layers,
            forward_expansion=train_forward_expansion,
            dropout=train_dropout,
            num_classes=num_classes,
            hidden_dim=train_hidden_dim,
            augment_eps=train_augment_eps
        ).to(device)

        model = load_model_weights(model, model_path, f"Model {i+1}")

        model.eval()
        with torch.no_grad():
            print(f"🔍 测试第{i+1}个模型...")
            global_probs, curr_true_labels, _, curr_test_residue_true, curr_test_residue_count = test_model(
                model, 
                test_loader, 
                test_window_metas, 
                test_total_residues,
                num_classes=num_classes,
                test_residue_labels_list=test_labels_windows
            )

        if true_labels_tensor is None:
            true_labels_tensor = curr_true_labels
            test_residue_true = curr_test_residue_true.cpu()  
            test_residue_count = curr_test_residue_count.cpu()  

        global_probs_corrected = residue_level_bayes_evidence(global_probs, prior=prior)
        global_probs_list.append(global_probs_corrected)

    fused_probs = torch.zeros_like(global_probs_list[0])
    for prob, weight in zip(global_probs_list, model_weights):
        fused_probs += weight * prob

    fused_probs_smoothed = smooth_scores_per_sequence(fused_probs, test_window_metas) 

    global_valid_mask = (test_residue_true[:, 0] != -1) & (test_residue_count > 0)
    valid_true_labels = test_residue_true[global_valid_mask].numpy()
    valid_fused_probs = fused_probs_smoothed[global_valid_mask].cpu().numpy()

    save_path = f"/root/autodl-tmp/MORF/Transformer/plot_data/TEST464.pkl"
    save_roc_data(valid_true_labels, valid_fused_probs, save_path)

    target_tprs = [0.2, 0.3, 0.4]
    fpr_results = get_fpr_at_fixed_tpr_morf(valid_true_labels, valid_fused_probs, target_tprs=target_tprs)

    fpr_save_path = f"/root/autodl-tmp/MORF/Transformer/results/TEST464-FPR.txt"
    save_fpr_results_to_txt_morf(fpr_results, filename=fpr_save_path)

    metrics_save_path = f"/root/autodl-tmp/MORF/Transformer/results/TEST464.txt"
    save_comprehensive_metrics(test_residue_true.numpy(), fused_probs_smoothed.cpu().numpy(), 
                               test_window_metas, metrics_save_path, model_weights)
    
    print("\n🎉 测试完成！")
    print(f"📌 结果保存路径：{save_path}")

# # ======================== MoRFchibi 2.0 extended datasets ========================
# if __name__ == "__main__":
#     print(f"当前配置：")
#     print(f"  逆贝叶斯校正: {ENABLE_BAYES_CORRECTION} (先验={BAYES_PRIOR_DEFAULT})")
#     print(f"  序列平滑: {ENABLE_SMOOTHING} (窗口={SMOOTH_WINDOW_SIZE})")
#     train_num_heads = 2  
#     train_num_layers = 4   
#     train_forward_expansion = 2  
#     train_dropout = 0.5    
#     train_hidden_dim = 256 
#     train_augment_eps = 0.05 

#     test_meta_path = "/root/autodl-tmp/MORF/Transformer/embedding/Test-106_success_metas.txt"
    
#     def load_memmap_data(feat_path, label_path, feat_dim=2304, num_classes=3):
#         feat_mmap = np.load(feat_path, mmap_mode='r')
#         label_mmap = np.load(label_path, mmap_mode='r')
#         return feat_mmap, label_mmap

#     test_encodings_np, test_labels_np = load_memmap_data(
#         "/root/autodl-tmp/MORF/Transformer/embedding/Test-106_success.npy",
#         "/root/autodl-tmp/MORF/Transformer/embedding/Test-106_success_label.npy",
#     )
#     test_window_metas = np.loadtxt(test_meta_path, dtype=np.int32)

#     test_total_residues = max(meta[4] for meta in test_window_metas) 
#     test_max_window_len = max(meta[4] - meta[3] for meta in test_window_metas)
    
#     print(f"测试集特征形状：{test_encodings_np.shape}")
#     print(f"测试集标签形状：{test_labels_np.shape}")
#     print(f"测试集窗口数：{len(test_window_metas)}")
#     print(f"测试集全局总残基数：{test_total_residues}")

#     test_encodings_windows, test_meta_indices = split_by_windows(test_encodings_np, test_window_metas, test_max_window_len)
#     test_labels_windows, _ = split_by_windows(test_labels_np, test_window_metas, test_max_window_len)

#     print(f"测试集窗口化后样本数：{len(test_encodings_windows)}")
#     print(f"测试集单个窗口特征形状：{test_encodings_windows[0].shape}")

#     test_dataset = ProteinDataset(test_encodings_windows, test_labels_windows, test_meta_indices)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     print("✅ Test loader loaded successfully")

#     model_paths = [
#         '/root/autodl-tmp/MORF/Transformer/model/final_cv1.pth',
#         '/root/autodl-tmp/MORF/Transformer/moedl/final_cv2.pth',
#         '/root/autodl-tmp/MORF/Transformer/moedl/final_cv3.pth',
#         '/root/autodl-tmp/MORF/Transformer/moedl/final_cv4.pth'
#     ]
#     priors_list = [0.1485, 0.1728, 0.1662, 0.1670]
#     model_weights = [0.25, 0.25, 0.25, 0.25] 
#     true_labels_tensor = None
#     global_probs_list = []
#     test_residue_true = None
#     test_residue_count = None
    
#     for i, (model_path, prior, weight) in enumerate(zip(model_paths, priors_list, model_weights)):
#         model = ProteinTransformer(
#             embed_dim=TOTAL_FEAT_DIM,
#             num_heads=train_num_heads,
#             num_layers=train_num_layers,
#             forward_expansion=train_forward_expansion,
#             dropout=train_dropout,
#             max_length=max_length,
#             num_classes=num_classes,
#             hidden_dim=train_hidden_dim,
#             augment_eps=train_augment_eps
#         ).to(device)

#         model = load_model_weights(model, model_path, f"Model {i+1}")

#         print(f"🔍 测试第{i+1}个模型...")
#         global_probs, curr_true_labels, _, curr_test_residue_true, curr_test_residue_count = test_model(
#             model, 
#             test_loader, 
#             test_window_metas, 
#             test_total_residues,
#             num_classes=num_classes,
#             test_residue_labels_list=test_labels_windows
#         )

#         if true_labels_tensor is None:
#             true_labels_tensor = curr_true_labels
#             test_residue_true = curr_test_residue_true
#             test_residue_count = curr_test_residue_count
#         global_probs_corrected = residue_level_bayes_evidence(global_probs, prior=prior)
#         global_probs_list.append(global_probs_corrected)

#     fused_probs = torch.zeros_like(global_probs_list[0])
#     for prob, weight in zip(global_probs_list, model_weights):
#         fused_probs += weight * prob

#     fused_probs_smoothed = smooth_scores_per_sequence(fused_probs, test_window_metas, window_size=SMOOTH_WINDOW_SIZE)

#     global_valid_mask = (test_residue_true[:, 0] != -1) & (test_residue_count > 0)
#     valid_true_labels = test_residue_true[global_valid_mask].cpu().numpy()
#     valid_fused_probs = fused_probs_smoothed[global_valid_mask].cpu().numpy()

#     save_path = f"/root/autodl-tmp/MORF/Transformer/plot_data/test4_smooth{SMOOTH_WINDOW_SIZE}_bayes{ENABLE_BAYES_CORRECTION}.pkl"
#     save_roc_data(valid_true_labels, valid_fused_probs, save_path)

#     target_tprs = [0.2, 0.3, 0.4]
#     fpr_results = get_fpr_at_fixed_tpr_morf(valid_true_labels, valid_fused_probs, target_tprs=target_tprs)

#     fpr_save_path = f"/root/autodl-tmp/MORF/Transformer/results/test4_FPR_smooth{SMOOTH_WINDOW_SIZE}_bayes{ENABLE_BAYES_CORRECTION}.txt"
#     save_fpr_results_to_txt_morf(fpr_results, filename=fpr_save_path)

#     metrics_save_path = f"/root/autodl-tmp/MORF/Transformer/results/test4_smooth{SMOOTH_WINDOW_SIZE}_bayes{ENABLE_BAYES_CORRECTION}.txt"
#     save_comprehensive_metrics(test_residue_true.cpu().numpy(), fused_probs_smoothed.cpu().numpy(), 
#                                test_window_metas, metrics_save_path, model_weights)
    
#     print("\n 测试完成！")
#     print(f"结果保存路径：{save_path}")