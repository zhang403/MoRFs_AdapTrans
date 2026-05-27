import os
import torch


shidi_basedir = "/root/autodl-tmp/datasets/shidi"
hj_basedir = "/root/autodl-tmp/MORF/Transformer/dataset"

def get_base_hj_dir():
    return hj_basedir
def get_base_shidi_dir():
    return shidi_basedir
def get_sub_hj_dir(dirname:str):
    target_path = os.path.join(hj_basedir, dirname)
    if not os.path.exists(target_path):
        raise FileNotFoundError
    return target_path
def get_sub_shidi_dir(dirname:str):
    target_path = os.path.join(shidi_basedir, dirname)
    if not os.path.exists(target_path):
        raise FileNotFoundError
    return target_path


def save_datasets_embedding_labels(embedding, labels, dataset_name, windows:list):
    windows = [str(i) for i in windows]
    save_dir = "/root/autodl-tmp/MORF/Transformer/dataset/tape"
    embedding_path = f"{dataset_name}_{'_'.join(windows)}_embedding.pt"
    labels_path = f"{dataset_name}_{'_'.join(windows)}_labels.pt"
    torch.save(embedding, os.path.join(save_dir, embedding_path))
    torch.save(labels, os.path.join(save_dir, labels_path))
    print(f"save embedding: {os.path.join(save_dir, embedding_path)}")
    print(f"save labels: {os.path.join(save_dir, labels_path)}")

def load_datasets_embedding_labels(dataset_name, windows:list):
    windows = [str(i) for i in windows]
    save_dir = "/root/autodl-tmp/MORF/Transformer/dataset/tape"
    embedding_path = f"{dataset_name}_{'_'.join(windows)}_embedding.pt"
    labels_path = f"{dataset_name}_{'_'.join(windows)}_labels.pt"
    print(f"load embedding: {os.path.join(save_dir, embedding_path)}")
    print(f"load labels: {os.path.join(save_dir, labels_path)}")
    return torch.load(os.path.join(save_dir, embedding_path)), torch.load(os.path.join(save_dir, labels_path))


def save_my_model(model, model_name:str, windows:list):
    windows = [str(i) for i in windows]
    import pickle
    savepath = f'{model_name}_{"_".join(windows)}.pkl'
    print(f"save my model: {savepath}")
    with open(savepath, "wb") as f:
        f.write(pickle.dumps(model))

def load_my_model(model, model_name:str, windows:list):
    windows = [str(i) for i in windows]
    import pickle
    savepath = f'{model_name}_{"_".join(windows)}.pkl'
    print(f"load my model: {savepath}")
    with open(savepath, "rb") as f:
        return pickle.loads(f.read())
