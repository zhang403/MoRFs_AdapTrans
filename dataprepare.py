import os
import torch
import numpy as np
import esm
from tqdm import tqdm
from typing import List, Tuple
from warnings import filterwarnings
filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_seqs_labels(dataset_path: str):
    seqs = []
    labels = []
    with open(dataset_path) as f:
        lines = f.readlines()
        group_num = len(lines) // 3
        assert group_num * 3 == len(lines)
        for i in range(group_num):
            seqs.append(lines[3 * i + 1].strip())
            labels.append(lines[3 * i + 2].strip())
    return seqs, labels

def get_sub_hj_dir(dirname: str):
    global hj_basedir
    hj_basedir = os.path.dirname(dirname) if os.path.dirname(dirname) else "."
    target_path = os.path.join(hj_basedir, os.path.basename(dirname))
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"子目录 {dirname} 不存在")
    return target_path

def merge_and_convert2int(labels):
    res = []
    for label in labels:
        for item in label:
            if item == '-':
                res.append(-1)
            else:
                res.append(int(item != "0"))
    return res

def get_prot_embeddings(sequences):
    from transformers import BertModel, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/MORF/Transformer/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("/root/autodl-tmp/MORF/Transformer/prot_bert").to(device)
    embedding = []
    for seq in tqdm(sequences):
        seq = " ".join(seq)
        encoded_input = tokenizer(seq, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(**encoded_input)[0][0]
        output = output[1:-1, :]
        embedding.append(output)
    print("embedding:", embedding[0].shape)
    return embedding

import esm
esmfold_model = esm.pretrained.esmfold_v1() 
esmfold_model = esmfold_model.eval().to(device)
esmfold_model.set_chunk_size(128)

def get_esm_embeddings(sequences, max_seq_len=1024):
    embeddings = []
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    batch_converter = alphabet.get_batch_converter()

    for seq in tqdm(sequences, desc="Extracting ESMFold Embeddings"):
        seq_len = len(seq)
        if seq_len <= max_seq_len:
            with torch.no_grad():
                data = [("protein", seq)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                esm_output = esmfold_model.esm(batch_tokens, repr_layers=[36], return_contacts=False)
                emb = esm_output["representations"][36].squeeze(0)[1:-1].cpu()
            embeddings.append(emb)
        else:
            num_chunks = (seq_len + max_seq_len - 1) // max_seq_len 
            chunk_embs = []
            for i in range(num_chunks):
                start = i * max_seq_len
                end = min((i+1) * max_seq_len, seq_len)
                sub_seq = seq[start:end]
                
                with torch.no_grad():
                    data = [("protein", sub_seq)]
                    _, _, batch_tokens = batch_converter(data)
                    batch_tokens = batch_tokens.to(device)
                    esm_output = esmfold_model.esm(batch_tokens, repr_layers=[36], return_contacts=False)
                    sub_emb = esm_output["representations"][36].squeeze(0)[1:-1].cpu()
                chunk_embs.append(sub_emb)

            full_emb = torch.cat(chunk_embs, dim=0)[:seq_len]
            embeddings.append(full_emb)
        
        del batch_tokens, esm_output
        torch.cuda.empty_cache()
    
    if embeddings:
        print(f"ESMFold Embedding dim: {embeddings[0].shape[1]}")
    return embeddings

def cut_sequence_windows(input_tensor, window_size, protein_id, global_residue_offset):
    input_tensor = input_tensor.to(device)
    seq_len, _ = input_tensor.shape
    seq_windows = []
    window_metas = []
    start_pos = 0

    while start_pos < seq_len:
        local_start = max(0, start_pos)
        local_end = min(seq_len, start_pos + window_size)
        global_start = global_residue_offset + local_start
        global_end = global_residue_offset + local_end
        seq_window = input_tensor[local_start:local_end, :]
        if seq_window.shape[0] > 0:
            seq_windows.append(seq_window)
            window_metas.append((protein_id, local_start, local_end, global_start, global_end))
        start_pos += window_size // 2

    return seq_windows, window_metas

def batch_cut_sequence_windows(encoding, window_size):
    all_seq_windows = []
    all_window_metas = []
    global_residue_offset = 0

    for protein_id, tensor in enumerate(encoding):
        seq_windows, window_metas = cut_sequence_windows(
            tensor, window_size, protein_id, global_residue_offset
        )
        all_seq_windows.extend(seq_windows)
        all_window_metas.extend(window_metas)
        global_residue_offset += tensor.shape[0]

    return all_seq_windows, all_window_metas, global_residue_offset

def apply_many_windows(encoding, windows):
    all_seq_windows = []
    all_window_metas = []
    total_residues = 0

    for window_size in windows:
        win_windows, win_metas, total_res = batch_cut_sequence_windows(encoding, window_size)
        all_seq_windows.extend(win_windows)
        all_window_metas.extend(win_metas)
        total_residues = total_res

    assert len(all_seq_windows) == len(all_window_metas), "序列片段与元数据数量不匹配"
    return all_seq_windows, all_window_metas, total_residues

def sync_labels_with_windows(labels, window_metas, total_residues):

    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
    assert labels_tensor.shape[0] == total_residues, f"标签长度({len(labels)})与总残基数({total_residues})不匹配"

    label_windows = []
    for meta in window_metas:
        _, _, _, global_start, global_end = meta
        label_window = labels_tensor[global_start:global_end, :]
        label_windows.append(label_window)

    return label_windows

def save_array_to_file(array, file_path):
    if array.ndim != 2:
        raise ValueError(f"输入必须是二维数组，当前维度：{array.ndim}")
    
    array = np.round(array, decimals=6)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, array, allow_pickle=False)

if __name__ == "__main__":
    dataset_Xtrain = "/root/autodl-tmp/MORF/Transformer/raw_seq2/Test-106_success.af"
    dataset_Xtrain_dir = get_sub_hj_dir(dataset_Xtrain)
    Xtrain_seqs, Xtrain_labels = get_seqs_labels(dataset_Xtrain_dir)
    Xtrian_embedding1 = get_prot_embeddings(Xtrain_seqs)
    Xtrian_embedding2 = get_esm_embeddings(Xtrain_seqs)
    Xtrian_embedding = []
    for prot_emb, esm_emb in zip(Xtrian_embedding1, Xtrian_embedding2):
        prot_emb = prot_emb.to(device)
        esm_emb = esm_emb.to(device)
        min_len = min(prot_emb.shape[0], esm_emb.shape[0])
        prot_emb = prot_emb[:min_len, :]
        esm_emb = esm_emb[:min_len, :]
        concat_emb = torch.cat([prot_emb, esm_emb], dim=1)
        Xtrian_embedding.append(concat_emb)

    apply_windows = [11]
    Xtrian_windows, window_metas, total_residues = apply_many_windows(Xtrian_embedding, apply_windows)
    Xtrain_single_label = merge_and_convert2int(Xtrain_labels)
    Xtrain_label_windows = sync_labels_with_windows(Xtrain_single_label, window_metas, total_residues)
    Xtrian_embedding_np = [tensor.cpu().numpy() for tensor in Xtrian_windows]
    Xtrian_embedding_np = np.concatenate(Xtrian_embedding_np, axis=0)
    save_path_emb = "/root/autodl-tmp/MORF/Transformer/embedding/Test-106_success_unmasked.npy"
    save_array_to_file(Xtrian_embedding_np, save_path_emb)
    Xtrain_label_np = [tensor.cpu().numpy() for tensor in Xtrain_label_windows]
    Xtrain_label_np = np.concatenate(Xtrain_label_np, axis=0)
    save_path_label = "/root/autodl-tmp/MORF/Transformer/embedding/Test-106_success_unmasked_label.npy"
    save_array_to_file(Xtrain_label_np, save_path_label)

    meta_np = np.array(window_metas, dtype=np.int32)
    save_path_meta = "/root/autodl-tmp/MORF/Transformer/embedding/Test-106_success_unmasked_metas.txt"
    np.savetxt(save_path_meta, meta_np, fmt="%d")
    print("数据处理流程结束！")
    
    
