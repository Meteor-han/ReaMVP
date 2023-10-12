# Overview

Data and codes for the paper "Prediction of Chemical Reaction Yields with Large-Scale Multi-View Pre-training"

[toc]

# Requirements

We implement our model on `Python 3.10`. These packages are mainly used:

```
rdkit			2022.9.5
torch			2.0.0+cu118
dgl				1.0.2+cu117
numpy			1.24.2
scikit-learn	1.2.2
```

# Datasets

1. Pre-training dataset

   We filtered reactions from USPTO and CJHIF. Related codes are stored in `data_utils`. You can download USPTO from https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873 and CJHIF from https://github.com/jshmjs45/data_for_chem.

2. Downstream dataset

   We finetuned our model on two downstream datasets. Related data for the Buchwald-Hartwig dataset and the Suzuki-Miyaura dataset is stored in `data/BH` and `data/SM`, respectively.

# Experiments

1. Pre-training

   Run `pretraining_stage1.py` and `pretraining_stage2.py` to pre-train ReaMVP in two stages, respectively. For example,

   ```
   python3 pretraining_stage1.py --seed 511 --device 0 --supervised 0 --epochs 30 --batch_size 256 --lr 0.001 --lr_type cos --T 1.0 --data_path ../data/pretraining_data/pretraining_cl
   
   python3 pretraining_stage2.py --seed 511 --device 0 --supervised 1 --data_type rnn --epochs 20 --batch_size 256 --lr 0.0001 --lr_type step --milestones 20 --save 0 --predictor_bn 0 --mlp_only 0 --loss_type mse --data_path ../data/pretraining_data/pretraining_yield
   ```

2. Fine-tuning

   Run `downstream/training.py` to fine-tune ReaMVP on a given downstream dataset. For example,

   ```
   python3 downstream/training.py --ds SM_test1 --device 0 --data_type rnn_geo --batch_size 128 --supervised 1 --lr 1e-3 --lr_type step --weight_decay 1e-5 --predictor_dropout 0.3 --predictor_num_layers 2 --gamma 0.3 --milestones 30 60 --epochs 90 --save 0 --normalize 0 --loss_type mse --cl_weight 1. --kl_weight 1. --predictor_bn 0 --repeat 10
   ```

   

