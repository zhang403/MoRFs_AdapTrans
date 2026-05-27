

import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.metrics import roc_curve, auc,roc_auc_score

pkl_base_dir = "/root/autodl-tmp/MORF/Transformer/plot_data/final" 
save_png_dir = "/root/autodl-tmp/MORF/Transformer/plot_data/final" 

def load_roc_pkl(path: str):
    with open(os.path.join(pkl_base_dir, path), "rb") as f:
        return pickle.load(f) 

def plot_roc(items):
    all_fpr = []
    all_tpr = []
    all_auc = []
    colors = []
    linestyles = []
    labels = [] 
    savepath = ""
    for item in items:
        true_labels, pred_score = load_roc_pkl(item[0])

        auc_value = roc_auc_score(true_labels, pred_score)
        all_auc.append(auc_value)

        fpr, tpr, _ = roc_curve(true_labels, pred_score)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        linestyles.append(item[1])
        colors.append(item[2])
        labels.append(item[3])
        savepath += item[3]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].set_xlim(0, 1)
    axs[0].set_xticks(np.arange(0, 1.1, 0.2))
    axs[0].set_ylim(0, 1)
    axs[0].set_yticks(np.arange(0, 1.15, 0.1))
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    for i, (fpr, tpr, auc_value) in enumerate(zip(all_fpr, all_tpr, all_auc)):
        axs[0].plot(fpr, tpr, linestyle=linestyles[i], color=colors[i], lw=1.2,
                    label=f'{labels[i]} ')
    axs[0].set_xlabel('FPR', fontsize=18)
    axs[0].set_ylabel('TPR', fontsize=18)
    axs[0].set_title('ROC Curves', fontsize=15)
    axs[0].legend(loc="lower right", fontsize=10)

    axs[1].set_xlim(0, 0.1)
    axs[1].set_xticks(np.arange(0, 0.11, 0.02))
    axs[1].set_ylim(0, 0.65)
    axs[1].set_yticks(np.arange(0, 0.66, 0.05))
    axs[1].tick_params(axis='both', which='major', labelsize=15)
    for i, (fpr, tpr, auc_value) in enumerate(zip(all_fpr, all_tpr, all_auc)):
        axs[1].plot(fpr, tpr, linestyle=linestyles[i], color=colors[i], lw=1,
                    label=f'{labels[i]}')   
    axs[1].set_xlabel('FPR', fontsize=18)
    axs[1].set_ylabel('TPR', fontsize=18)
    axs[1].set_title('ROC Curves in the Low FPR Area', fontsize=15)
    axs[1].legend(loc="lower right", fontsize=10)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(os.path.join(save_png_dir, savepath + ".png"))

    plt.show()  

def main():
    items = [  
        ["plot_data/module_ablation/baseline.pkl", "-", "brown", "Baseline"],
        ["plot_data/module_ablation/baseline_AFF.pkl", "-", "grey", "Baseline+AFF"],
        ["plot_data/module_ablation/baseline_GLA.pkl", "-", "pink", "Baseline+GLA"],
        ["plot_data/module_ablation/baseline_CAF.pkl", "-", "purple", "Baseline+CAF"],
        ["plot_data/module_ablation/baseline_AFF_GLA.pkl", "-", "yellow", "Baseline+AFF+GLA"],
        ["plot_data/module_ablation/baseline_AFF_CAF.pkl", "-", "green", "Baseline+AFF+CAF"],
        ["plot_data/module_ablation/baseline_GLA_CAF.pkl", "-", "blue", "Baseline+GLA+CAF"],
        ["plot_data/traditional_benchmark_datasets/test464-0.862.pkl", "-", "red", "Baseline+AFF+GLA+CA"]

    ]
    plot_roc(items)
    
if __name__ == "__main__":
    main()
