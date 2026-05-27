import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


batch_size = 32 
num_epochs = 100  
patience = 3 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 1  
max_length = 11

PROT_FEAT_DIM = 1024 
ESM_FEAT_DIM = 2560   
TOTAL_FEAT_DIM = PROT_FEAT_DIM + ESM_FEAT_DIM

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
            local_window_sizes=[3,5], 
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

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1.5, gamma_neg=5.0, prob_margin=0.1, class_weight=3.0, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.eps = eps
        self.gamma_pos = gamma_pos  
        self.gamma_neg = gamma_neg  
        self.prob_margin = prob_margin  
        self.class_weight = class_weight

    def forward(self, inputs, targets):
        if inputs.dim() == 2: inputs = inputs.squeeze(-1)
        if targets.dim() == 2: targets = targets.squeeze(-1)
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        probabilities = torch.sigmoid(inputs)
        pos_mask = (targets == 1).float()
        pos_term = 1.0 - probabilities
        pos_loss = -torch.pow(pos_term, self.gamma_pos) * torch.log(probabilities + self.eps)
        pos_loss = pos_mask * pos_loss * self.class_weight
        neg_mask = (targets == 0).float()
        shifted_prob = torch.clamp(probabilities - self.prob_margin, min=0.0, max=1.0)
        neg_term = 1.0 - shifted_prob
        neg_loss = -torch.pow(shifted_prob, self.gamma_neg) * torch.log(neg_term + self.eps)
        neg_loss = neg_mask * neg_loss
        
        return torch.mean(pos_loss + neg_loss)

    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5,
                train_total_residues=None, train_window_metas=None, val_total_residues=None, val_window_metas=None,
                train_residue_labels_list=None, val_residue_labels_list=None):
    print("start train model (segment-level loss for MoRFs)")
    best_val_auc = -float('inf')
    counter = 0
    best_model_weights = None
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_segment_preds = []
        train_segment_labels = []

        for i, (inputs, segment_true_labels, meta_indices) in enumerate(train_loader):
            inputs, segment_true_labels = inputs.to(device), segment_true_labels.to(device).float()
            optimizer.zero_grad()

            segment_preds = model(inputs)
            loss = criterion(segment_preds, segment_true_labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_segment_preds.append(segment_preds.detach().cpu())
            train_segment_labels.append(segment_true_labels.cpu())

        train_segment_preds = torch.cat(train_segment_preds, dim=0)
        train_segment_labels = torch.cat(train_segment_labels, dim=0)
        train_segment_probs = torch.sigmoid(train_segment_preds)
        train_segment_results = compute_metrics_per_residue_detailed(
            train_segment_labels,
            train_segment_probs
        )
        train_segment_auc = train_segment_results['total_auc']
        train_loss_avg = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_segment_preds = []
        val_segment_labels = []
        val_residue_prob_sum = torch.zeros((val_total_residues, num_classes), dtype=torch.float32, device=device)
        val_residue_count = torch.zeros(val_total_residues, dtype=torch.int32, device=device)
        val_residue_true = torch.zeros((val_total_residues, num_classes), dtype=torch.float32, device=device)

        with torch.no_grad():
            for i, (inputs, segment_true_labels, meta_indices) in enumerate(val_loader):
                inputs, segment_true_labels = inputs.to(device), segment_true_labels.to(device).float()
                segment_preds = model(inputs)
                loss = criterion(segment_preds, segment_true_labels)
                val_loss += loss.item()

                val_segment_preds.append(segment_preds.cpu())
                val_segment_labels.append(segment_true_labels.cpu())

                segment_probs = torch.sigmoid(segment_preds)
                for idx in range(len(inputs)):
                    meta_idx = meta_indices[idx].item()
                    meta = val_window_metas[meta_idx]
                    global_start, global_end = meta[3], meta[4]
                    actual_len = global_end - global_start

                    residue_probs = segment_probs[idx].unsqueeze(0).repeat(actual_len, 1)
                    val_residue_prob_sum[global_start:global_end, :] += residue_probs
                    val_residue_count[global_start:global_end] += 1
                    val_residue_true[global_start:global_end, :] = val_residue_labels_list[meta_idx][:actual_len].clone().detach()

        val_segment_preds = torch.cat(val_segment_preds, dim=0)
        val_segment_labels = torch.cat(val_segment_labels, dim=0)
        val_segment_probs = torch.sigmoid(val_segment_preds)
        val_segment_results = compute_metrics_per_residue_detailed(
            val_segment_labels,
            val_segment_probs
        )
        val_segment_auc = val_segment_results['total_auc']
        val_loss_avg = val_loss / len(val_loader)

        val_global_avg_probs = val_residue_prob_sum / val_residue_count.unsqueeze(1).clamp(min=1)
        label_valid_mask = (val_residue_true != -1).squeeze(-1)
        count_valid_mask = (val_residue_count > 0)
        val_mask = label_valid_mask & count_valid_mask
        val_mask = val_mask.unsqueeze(1).expand(-1, num_classes)
        
        val_masked_probs = val_global_avg_probs[val_mask].reshape(-1, num_classes)
        val_masked_labels = val_residue_true[val_mask].reshape(-1, num_classes)
        val_residue_results = compute_metrics_per_residue_detailed(
            val_masked_labels,
            val_masked_probs
        )
        val_residue_auc = val_residue_results['total_auc']

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train - Loss: {train_loss_avg:.4f}, segment-AUC: {train_segment_auc:.4f}")
        print(f"  Val   - Loss: {val_loss_avg:.4f}, segment-AUC: {val_segment_auc:.4f}, Residue-AUC: {val_residue_auc:.4f}")
        print("  MoRFs Val Metrics:")
        metrics = val_residue_results['per_label']['MoRFs']
        print(f"    Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        print()

        if val_residue_auc > best_val_auc:
            best_val_auc = val_residue_auc
            best_model_weights = model.state_dict()
            counter = 0  
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (残基AUC连续{patience}轮未提升)")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    return model, best_val_auc  


def compute_metrics_per_residue_detailed(labels, outputs):
    if isinstance(outputs, torch.Tensor):
        if torch.abs(outputs).max() > 10:
            outputs_prob = torch.sigmoid(outputs).detach().cpu().numpy()
        else:
            outputs_prob = outputs.detach().cpu().numpy()
    else:
        outputs_prob = np.array(outputs)
    
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)

    outputs_pred = (outputs_prob >= 0.5).astype(int)
    if labels_np.ndim == 1:
        labels_np = labels_np.reshape(-1, 1)
    if outputs_pred.ndim == 1:
        outputs_pred = outputs_pred.reshape(-1, 1)
    if outputs_prob.ndim == 1:
        outputs_prob = outputs_prob.reshape(-1, 1)

    per_label_metrics = {}
    label_name = "MoRFs"
    label_idx = 0  

    try:
        tn, fp, fn, tp = confusion_matrix(labels_np[:, label_idx], outputs_pred[:, label_idx], labels=[0, 1]).ravel()
    except Exception as e:
        tn, fp, fn, tp = 0, 0, 0, 0
        if np.all(labels_np[:, label_idx] == 0):
            tn = len(labels_np[:, label_idx])
            fp = np.sum(outputs_pred[:, label_idx] == 1)
        elif np.all(labels_np[:, label_idx] == 1):
            tp = len(labels_np[:, label_idx])
            fn = np.sum(outputs_pred[:, label_idx] == 0)
        else:
            print(f"混淆矩阵计算异常: {e}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = recall 
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        valid_label_mask = (labels_np[:, label_idx] == 0) | (labels_np[:, label_idx] == 1)
        auc = roc_auc_score(
            labels_np[valid_label_mask, label_idx],
            outputs_prob[valid_label_mask, label_idx]
        )
    except ValueError as e:
        auc = float('nan')
        print(f"AUC计算异常: {e}")

    per_label_metrics[label_name] = {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'auc': auc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

    total_accuracy = accuracy_score(labels_np[:, label_idx], outputs_pred[:, label_idx])
    total_precision = precision
    total_recall = recall
    total_f1 = f1
    total_tpr = tpr
    total_fpr = fpr
    total_tnr = tnr
    total_auc = auc

    return {
        'per_label': per_label_metrics,
        'total_accuracy': total_accuracy, 'total_precision': total_precision,
        'total_recall': total_recall, 'total_f1': total_f1,
        'total_tpr': total_tpr, 'total_fpr': total_fpr, 'total_tnr': total_tnr,
        'total_auc': total_auc
    }
# ======================== Traditional benchmark datasets ========================
if __name__ == "__main__":
    base_path = "/root/autodl-tmp/MORF/Transformer/embedding/"
    save_dir = "/root/autodl-tmp/MORF/Transformer/model/"
    os.makedirs(save_dir, exist_ok=True)
    data_files = {
        "train": {
            "feat": os.path.join(base_path, "Train.npy"),
            "label": os.path.join(base_path, "Train_label.npy"),
            "meta": os.path.join(base_path, "Train_metas.txt")
        }
    }
    
    def load_memmap_data(feat_path, label_path):
        feat_mmap = np.load(feat_path, mmap_mode='r')
        label_mmap = np.load(label_path, mmap_mode='r')
        return feat_mmap, label_mmap

    full_encodings_np, full_labels_np = load_memmap_data(
        data_files["train"]["feat"], 
        data_files["train"]["label"]
    )
    full_window_metas = np.loadtxt(data_files["train"]["meta"], dtype=np.int32)

    print("\n===================== 按序列8:2分割训练集/验证集 =====================")
    unique_seq_ids = np.unique(full_window_metas[:, 0])
    random_seed = 42
    
    train_seq_ids, val_seq_ids = train_test_split(
        unique_seq_ids,
        test_size=0.2,
        random_state=random_seed,
        shuffle=True
    )

    train_window_metas = full_window_metas[np.isin(full_window_metas[:, 0], train_seq_ids)]
    val_window_metas = full_window_metas[np.isin(full_window_metas[:, 0], val_seq_ids)]

    train_max_window_len = max(meta[4] - meta[3] for meta in train_window_metas)
    val_max_window_len = max(meta[4] - meta[3] for meta in val_window_metas)
    max_window_len = max(train_max_window_len, val_max_window_len)
    
    print(f"训练集窗口数：{len(train_window_metas)}")
    print(f"验证集窗口数：{len(val_window_metas)}")
    print(f"统一最大窗口长度：{max_window_len}")
    
    train_encodings_windows, train_meta_indices = split_by_windows(
        full_encodings_np, train_window_metas, max_window_len
    )
    train_labels_windows, _ = split_by_windows(
        full_labels_np, train_window_metas, max_window_len
    )
    val_encodings_windows, val_meta_indices = split_by_windows(
        full_encodings_np, val_window_metas, max_window_len
    )
    val_labels_windows, _ = split_by_windows(
        full_labels_np, val_window_metas, max_window_len
    )
    train_total_residues = max(meta[4] for meta in train_window_metas)
    val_total_residues = max(meta[4] for meta in val_window_metas)
    train_dataset = ProteinDataset(train_encodings_windows, train_labels_windows, train_meta_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ProteinDataset(val_encodings_windows, val_labels_windows, val_meta_indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model_instances = []  
    num_instances = 15   
    
    for inst in range(num_instances):
        print(f"\n----- 训练第 {inst+1}/{num_instances} 个模型实例 -----")
        model = ProteinTransformer(
            embed_dim=train_encodings_windows[0].shape[1],
            num_heads=2,
            num_layers=4,
            forward_expansion=2,
            dropout=0.6,
            hidden_dim=256,
            augment_eps=0.05
        ).to(device)
        
        criterion = AsymmetricLoss(
            gamma_pos=4.0,
            gamma_neg=2.8,
            prob_margin=0.2,
            class_weight=2.5,
            eps=1e-8
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
        
        trained_model, best_auc = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            patience=patience,
            train_total_residues=train_total_residues,
            train_window_metas=train_window_metas,
            val_total_residues=val_total_residues,
            val_window_metas=val_window_metas,
            val_residue_labels_list=val_labels_windows
        )
        
        inst_save_path = os.path.join(save_dir, f"inst{inst+1}_auc{best_auc:.4f}.pth")
        torch.save(trained_model.state_dict(), inst_save_path)
        model_instances.append((best_auc, inst_save_path))
        print(f"模型实例{inst+1}保存完成：{inst_save_path} (AUC={best_auc:.4f})")
        
    model_instances.sort(reverse=True, key=lambda x: x[0])
    third_best_auc, third_best_path = model_instances[2]  # 索引2对应排名第三
    print(f"\n训练完成：选择AUC排名第三的模型 - AUC={third_best_auc:.4f}，路径={third_best_path}")
    final_model_path = os.path.join(save_dir, "final_best_model.pth")
    torch.save(torch.load(third_best_path), final_model_path)
    print(f"最终模型保存完成：{final_model_path}")

    print("\n===================== 训练全部完成 =====================")
    print(f"最终模型文件已保存至：{final_model_path}")
    print(f"所有模型实例均保存在：{save_dir}")


# # ======================== MoRFchibi 2.0 extended datasets ========================
# if __name__ == "__main__":
#     base_path = "/root/autodl-tmp/MORF/Transformer/embedding/"
#     save_dir = "/root/autodl-tmp/MORF/Transformer/model/"
#     os.makedirs(save_dir, exist_ok=True)

#     cv_files = {
#         0: {
#             "feat": os.path.join(base_path, "CV1.npy"),
#             "label": os.path.join(base_path, "CV1_label.npy"),
#             "meta": os.path.join(base_path, "CV1_metas.txt")
#         },
#         1: {
#             "feat": os.path.join(base_path, "CV2.npy"),
#             "label": os.path.join(base_path, "CV2_label.npy"),
#             "meta": os.path.join(base_path, "CV2_metas.txt")
#         },
#         2: {
#             "feat": os.path.join(base_path, "CV3.npy"),
#             "label": os.path.join(base_path, "CV3_label.npy"),
#             "meta": os.path.join(base_path, "CV3_metas.txt")
#         },
#         3: {
#             "feat": os.path.join(base_path, "CV4.npy"),
#             "label": os.path.join(base_path, "CV4_label.npy"),
#             "meta": os.path.join(base_path, "CV4_metas.txt")
#         }
#     }

#     def load_memmap_data(feat_path, label_path):
#         feat_mmap = np.load(feat_path, mmap_mode='r')
#         label_mmap = np.load(label_path, mmap_mode='r')
#         return feat_mmap, label_mmap

#     for fold in range(4):
#         print(f"\n===================== 开始第 {fold+1} 折CV训练 =====================")
#         train_folds = [i for i in range(4) if i != fold]
#         val_fold = fold
#         train_feats_list = []
#         train_labels_list = []
#         train_metas_list = []
#         train_max_window_len = 0
#         train_total_residues = 0
        
#         for tf in train_folds:
#             tf_feat, tf_label = load_memmap_data(cv_files[tf]["feat"], cv_files[tf]["label"])
#             tf_metas = np.loadtxt(cv_files[tf]["meta"], dtype=np.int32)
#             train_feats_list.append(tf_feat)
#             train_labels_list.append(tf_label)
#             train_metas_list.extend(tf_metas)
#             tf_window_len = max(meta[4] - meta[3] for meta in tf_metas)
#             tf_total_res = max(meta[4] for meta in tf_metas)
#             train_max_window_len = max(train_max_window_len, tf_window_len)
#             train_total_residues += tf_total_res

#         train_encodings_np = np.concatenate(train_feats_list, axis=0)
#         train_labels_np = np.concatenate(train_labels_list, axis=0)
#         train_window_metas = np.array(train_metas_list)

#         val_encodings_np, val_labels_np = load_memmap_data(cv_files[val_fold]["feat"], cv_files[val_fold]["label"])
#         val_window_metas = np.loadtxt(cv_files[val_fold]["meta"], dtype=np.int32)
#         val_max_window_len = max(meta[4] - meta[3] for meta in val_window_metas)
#         val_total_residues = max(meta[4] for meta in val_window_metas)
        
#         print(f"训练集：合并{len(train_folds)}折，特征形状={train_encodings_np.shape}，标签形状={train_labels_np.shape}")
#         print(f"验证集：第{val_fold+1}折，特征形状={val_encodings_np.shape}，标签形状={val_labels_np.shape}")
#         train_encodings_windows, train_meta_indices = split_by_windows(train_encodings_np, train_window_metas, train_max_window_len)
#         train_labels_windows, _ = split_by_windows(train_labels_np, train_window_metas, train_max_window_len)
        
#         val_encodings_windows, val_meta_indices = split_by_windows(val_encodings_np, val_window_metas, val_max_window_len)
#         val_labels_windows, _ = split_by_windows(val_labels_np, val_window_metas, val_max_window_len)
#         train_dataset = ProteinDataset(train_encodings_windows, train_labels_windows, train_meta_indices)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
#         val_dataset = ProteinDataset(val_encodings_windows, val_labels_windows, val_meta_indices)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#         model_instances = [] 
#         for inst in range(10):
#             print(f"\n----- 训练第 {inst+1}/15 个模型实例 -----")
#             model = ProteinTransformer(
#                 embed_dim=train_encodings_windows[0].shape[1],  
#                 num_heads=2,
#                 num_layers=4,
#                 forward_expansion=2,
#                 dropout=0.6,
#                 hidden_dim=256,
#                 augment_eps=0.05
#             ).to(device)
#             criterion = AsymmetricLoss(
#                 gamma_pos=4.0,
#                 gamma_neg=2.8,
#                 prob_margin=0.2,
#                 class_weight=2.5,
#                 eps=1e-8
#             ).to(device)
            
#             optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
#             trained_model, best_auc = train_model(
#                 model,
#                 train_loader,
#                 val_loader,
#                 criterion,
#                 optimizer,
#                 num_epochs=num_epochs,
#                 patience=patience,
#                 train_total_residues=train_total_residues,
#                 train_window_metas=train_window_metas,
#                 val_total_residues=val_total_residues,
#                 val_window_metas=val_window_metas,
#                 val_residue_labels_list=val_labels_windows
#             )
#             inst_save_path = os.path.join(save_dir, f"cv{fold+1}_inst{inst+1}_auc{best_auc:.4f}.pth")
#             torch.save(trained_model.state_dict(), inst_save_path)
#             model_instances.append((best_auc, inst_save_path))
#             print(f"模型实例{inst+1}保存完成：{inst_save_path} (AUC={best_auc:.4f})")
#         model_instances.sort(reverse=True, key=lambda x: x[0])
#         third_best_auc, third_best_path = model_instances[2] 
#         print(f"\n第{fold+1}折CV：选择AUC排名第三的模型 - AUC={third_best_auc:.4f}，路径={third_best_path}")
#         final_model_path = os.path.join(save_dir, f"final_cv{fold+1}.pth")
#         torch.save(torch.load(third_best_path), final_model_path)
#         print(f"第{fold+1}折CV最终模型保存完成：{final_model_path}")
#     print("\n===================== 四折CV训练全部完成 =====================")
#     print(f"四个最终模型文件已保存至：{save_dir}")
#     print("模型文件列表：")
#     for fold in range(4):
#         print(f"  - final_cv{fold+1}.pth")