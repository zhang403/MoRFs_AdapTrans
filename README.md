# MoRFs_AdapTrans
MoRFs_AdapTrans：MoRFs Prediction Based on Adaptive Feature Fusion and Improved Transformer network
## Model Overview
<img width="1954" height="828" alt="image" src="https://github.com/user-attachments/assets/d1a8c9ee-657b-47b3-b1e5-e449c91b0dd4" />
## How to Use
Install the dependencies for ProtBERT and ESMFold pre-trained models from official websites first.
Add your training and testing dataset in the data directory or use pre-prepared datasets.
Then, run the dataprepare.py file to obtain the embeddings.
```bash
python dataprepare.py
Next, run the train.py file to train the MoRFs_AdapTrans model:
```bash
python train.py
Note that we used two different datasets in our paper, so there are two sets of training code. If you want to train your own model, please select the corresponding one according to your dataset format.
Finally, run the test.py file to perform tests：
```bash
python test.py
Note: We provide our trained models in the `model` folder, including those for various ablation experiments, traditional benchmark datasets, and MoRFchibi 2.0 extended datasets. You can directly load these models to test on your own datasets.






