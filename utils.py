import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl
import shutil
import random
import logging
import argparse
from typing import List, Union
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')
data_prefix = "../../data/shirunhan/reaction"


# arguments
def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--log_path', default="log0", type=str, help="log path")

    parser.add_argument('--seed', type=int, default=217)
    parser.add_argument('--device', type=int, default=0, choices=[i for i in range(8)])
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--data_workers', type=int, default=4)
    parser.add_argument('--supervised', type=int, default=0,
                        choices=[0, 1, 2], help="0: self-supervised, 1: then supervised, 2: supervised only")
    parser.add_argument('--data_type', type=str, default="rnn_geo",
                        choices=["rnn", "geo", "rnn_geo"])
    parser.add_argument('--normalize', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss_type', type=str, default="mse", choices=["mse", "mae"])
    parser.add_argument('--mlp_only', type=int, default=0, choices=[0, 1], help="train/finetune mlp only or not")

    # pretraining dataset, 0 for cl, 1 for yield
    parser.add_argument('--data_path',
                        default=os.path.join(data_prefix, "data", "pretraining_data", "pretraining_cl"), type=str)
    parser.add_argument('--pos_dict_path',
                        default=os.path.join(data_prefix, "data", "pretraining_data", "smiles_pos_dict.pt"), type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    # downstream dataset
    parser.add_argument("--ds", default="None", type=str,
                        choices=["BH", "SM", "SM_test1", "SM_test2", "SM_test3", "SM_test4",
                                 "BH_test1", "BH_test2", "BH_test3", "BH_test4",
                                 "BH_plate1", "BH_plate2", "BH_plate3", "BH_plate2_new"])
    parser.add_argument("--repeat", default=10, type=int)
    parser.add_argument("--split", type=float, nargs='+', default=[0.7])

    # training strategy
    parser.add_argument("--lr_type", type=str, default="step", choices=["step", "cos"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, nargs='+', default=[10, 30])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--cos_rate", type=float, default=0.1)
    parser.add_argument("--projector_lr_scale", type=float, default=1.0)
    parser.add_argument("--predictor_lr_scale", type=float, default=1.0)
    parser.add_argument("--predictor_dropout", type=float, default=0.1)
    parser.add_argument("--predictor_bn", type=int, default=0)
    parser.add_argument("--predictor_num_layers", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--metric", type=str, default="InfoNCE_dot_prod",
                        choices=["InfoNCE_dot_prod", "EBM_dot_prod"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument('--T', type=float, default=0.1, help="Temperature for CL")
    parser.add_argument('--cl_weight', type=float, default=1., help="CL loss weight")
    parser.add_argument('--kl_weight', type=float, default=-1., help="KL loss weight")

    # SMILESRNN model
    parser.add_argument("--smiles_embed_size", type=int, default=256)
    parser.add_argument("--smiles_hidden_dim", type=int, default=128)
    parser.add_argument("--smiles_lr_scale", type=float, default=1.0)
    parser.add_argument("--smiles_dropout", type=float, default=0.3)
    parser.add_argument("--smiles_use_lstm", type=int, default=1)
    parser.add_argument("--smiles_use_bidirectional", type=int, default=1)
    parser.add_argument("--smiles_n_layer", type=int, default=2)
    # GNN/GeoRNN
    parser.add_argument("--rnn_hidden_dim", type=int, default=128)
    parser.add_argument("--rnn_lr_scale", type=float, default=1.0)
    parser.add_argument("--rnn_dropout", type=float, default=0.3)
    parser.add_argument("--rnn_use_lstm", type=int, default=1)
    parser.add_argument("--rnn_use_bidirectional", type=int, default=1)
    parser.add_argument("--rnn_n_layer", type=int, default=2)
    # Geo
    parser.add_argument("--sch_n_interaction", type=int, default=4)
    parser.add_argument("--sch_hidden_dim", type=int, default=128)
    parser.add_argument("--sch_n_filter", type=int, default=128)
    parser.add_argument("--sch_n_gaussian", type=int, default=64)
    parser.add_argument("--sch_cutoff", type=float, default=10.0)
    parser.add_argument("--geo_lr_scale", type=float, default=1.0)

    args = parser.parse_args()

    args.smiles_use_lstm = True if args.smiles_use_lstm == 1 else False
    args.smiles_use_bidirectional = True if args.smiles_use_bidirectional == 1 else False
    args.rnn_use_lstm = True if args.rnn_use_lstm == 1 else False
    args.rnn_use_bidirectional = True if args.rnn_use_bidirectional == 1 else False
    return args


def mol_to_dgl_graph(mol, use_3d=True, pos=None, radius=10.0):
    if use_3d:
        pos = torch.as_tensor(pos, dtype=torch.float)
    else:
        try:
            mol = Chem.RemoveHs(mol)
        except:
            pass

    # prepare node features
    n_atoms = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = np.array(get_atom_features(atom))
        features.append(feature / sum(feature))  # norm

    # construct a dgl graph from radius
    if use_3d and radius is not None:
        g = dgl.radius_graph(pos, radius, self_loop=True)
    # construct a dgl graph from bonds
    else:
        g = dgl.graph(([], []))
        g.add_nodes(n_atoms)
        edge_out = []
        edge_in = []
        for bond in mol.GetBonds():  # edge as [i, j]
            edge_out.append(bond.GetBeginAtomIdx())
            edge_in.append(bond.GetEndAtomIdx())
        g = dgl.add_edges(g, edge_out, edge_in)
        # isolated graph
        g = dgl.add_self_loop(g)
    if g.number_of_nodes() != n_atoms:
        print("isolation occurs")
        print(Chem.MolToSmiles(mol), g.number_of_nodes(), n_atoms)
        return None
    g.ndata['feature'] = torch.as_tensor(np.array(features), dtype=torch.float)
    if use_3d:
        g.ndata['position'] = pos

    return g


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    param value: The value for which the encoding should be one.
    param choices: A list of possible values.
    return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def get_atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    param atom: An RDKit atom.
    param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    return: A list containing the atom features.
    """
    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES = {
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-1, -2, 1, 2, 0],
        'chiral_tag': [0, 1, 2, 3],
        'num_Hs': [0, 1, 2, 3, 4],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ],
    }
    ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
    if atom is None:
        features = [0] * ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
                   onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
                   onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
                   onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
                   onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
                   onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
                   [1 if atom.GetIsAromatic() else 0] + \
                   [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return features


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# finetune either 1d or 3d, split the dataloader. memory?
def my_collate_fn_3d(data):
    # [[keys, one_index_list, one_dgl_list, y], ...]
    dgl_pool, dgl_len_pool, target_pool = [], [], []
    smiles_index_pool, smiles_index_len_pool = [], []
    for keys, one_index_list, one_dgl_list, y in data:
        smiles_index_pool.append(torch.tensor(one_index_list, dtype=torch.long))
        smiles_index_len_pool.append(len(one_index_list))
        for (_, dgl_3d) in one_dgl_list:
            dgl_pool.append(dgl_3d)
        dgl_len_pool.append(len(one_dgl_list))
        target_pool.append(y)
    return rnn_utils.pad_sequence(smiles_index_pool, batch_first=True, padding_value=0), \
           torch.tensor(smiles_index_len_pool), \
           dgl.batch(dgl_pool), torch.tensor(dgl_len_pool), torch.tensor(target_pool, dtype=torch.float).unsqueeze(-1)


def get_data_loader_cl(dataset, pos_dict, use_3d=True, radius=10.0, batch_size=256):
    dgl_pool, dgl_len_pool = [], []
    smiles_index_pool, smiles_index_len_pool = [], []
    cnt = 0
    for idd, (l, c, r, reaction, one_index_list) in enumerate(dataset):
        smiles_index_pool.append(torch.tensor(one_index_list, dtype=torch.long))
        smiles_index_len_pool.append(len(one_index_list))
        for s in reaction:
            dgl_pool.append(mol_to_dgl_graph(Chem.MolFromSmiles(s), use_3d, pos_dict[s], radius))
        dgl_len_pool.append(len(reaction))
        cnt += 1
        if cnt == batch_size:
            yield rnn_utils.pad_sequence(smiles_index_pool, batch_first=True, padding_value=0), \
                  torch.tensor(smiles_index_len_pool), \
                  dgl.batch(dgl_pool), torch.tensor(dgl_len_pool)
            dgl_pool, dgl_len_pool = [], []
            smiles_index_pool, smiles_index_len_pool = [], []
            cnt = 0
    if len(dgl_pool) > 0:
        yield rnn_utils.pad_sequence(smiles_index_pool, batch_first=True, padding_value=0), \
              torch.tensor(smiles_index_len_pool), \
              dgl.batch(dgl_pool), torch.tensor(dgl_len_pool)


def get_data_loader_y(dataset, pos_dict, use_3d=True, radius=10.0, batch_size=256, geo=False):
    dgl_pool, dgl_len_pool, target_pool = [], [], []
    smiles_index_pool, smiles_index_len_pool = [], []
    cnt = 0
    for idd, (l, c, r, reaction, one_index_list, y) in enumerate(dataset):
        smiles_index_pool.append(torch.tensor(one_index_list, dtype=torch.long))
        smiles_index_len_pool.append(len(one_index_list))
        if geo:
            # still contains ""?
            temp_n = len(reaction)
            for s in reaction:
                if not s:
                    temp_n -= 1
                    continue
                dgl_pool.append(mol_to_dgl_graph(Chem.MolFromSmiles(s), use_3d, pos_dict[s], radius))
            dgl_len_pool.append(temp_n)
        # not do forget... 100
        target_pool.append(y / 100)
        cnt += 1
        if cnt == batch_size:
            if not geo:
                dgl_pool.append(dgl.graph(([0], [1])))
                dgl_len_pool.append(1)
            yield rnn_utils.pad_sequence(smiles_index_pool, batch_first=True, padding_value=0), \
                  torch.tensor(smiles_index_len_pool), \
                  dgl.batch(dgl_pool), torch.tensor(dgl_len_pool), \
                  torch.tensor(target_pool, dtype=torch.float).unsqueeze(-1)
            dgl_pool, dgl_len_pool, target_pool = [], [], []
            smiles_index_pool, smiles_index_len_pool = [], []
            cnt = 0
    if len(smiles_index_pool) > 0:
        if not geo:
            dgl_pool.append(dgl.graph(([0], [1])))
            dgl_len_pool.append(1)
        yield rnn_utils.pad_sequence(smiles_index_pool, batch_first=True, padding_value=0), \
              torch.tensor(smiles_index_len_pool), \
              dgl.batch(dgl_pool), torch.tensor(dgl_len_pool), \
              torch.tensor(target_pool, dtype=torch.float).unsqueeze(-1)


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-3] + "_best.pt")


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, normalize, metric, T, lambda_cl, lambda_kl, CL_neg_samples):
    """
    :param X: "input"
    :param Y: "target"
    :param normalize: bool, l2 or not
    :param metric: InfoNCE
    :param T: temperature
    :param lambda_cl: loss, weight
    :param lambda_kl: loss, weight
    :param CL_neg_samples:
    :return:
    """
    # default, l2 normalization
    if normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)
    CL_loss, KL_loss = torch.tensor(0.), torch.tensor(0.)
    if lambda_cl > 0.:
        if metric == 'InfoNCE_dot_prod':
            criterion = nn.CrossEntropyLoss()
            B = X.size()[0]
            logits = torch.mm(X, Y.transpose(1, 0))  # B*B
            logits = torch.div(logits, T)
            labels = torch.arange(B, dtype=torch.long).to(logits.device)  # B*1

            CL_loss = criterion(logits, labels)
            # pred = logits.argmax(dim=1, keepdim=False)
            # CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

        elif metric == 'EBM_dot_prod':
            criterion = nn.BCEWithLogitsLoss()
            neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)]
                               for i in range(CL_neg_samples)], dim=0)
            neg_X = X.repeat((CL_neg_samples, 1))

            pred_pos = torch.sum(X * Y, dim=1) / T
            pred_neg = torch.sum(neg_X * neg_Y, dim=1) / T

            loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
            loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
            CL_loss = loss_pos + CL_neg_samples * loss_neg

            # CL_acc = (torch.sum(pred_pos > 0).float() +
            #           torch.sum(pred_neg < 0).float()) / \
            #          (len(pred_pos) + len(pred_neg))
            # CL_acc = CL_acc.detach().cpu().item()

        else:
            raise Exception

    # apply KL-divergence
    if lambda_kl > 0.:
        kl_criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
        KL_loss = kl_criterion(F.log_softmax(X, dim=-1), F.log_softmax(Y, dim=-1))

    return CL_loss, KL_loss


def dual_CL(X, Y, normalize=True, metric="InfoNCE_dot_prod", T=0.1, lambda_cl=1., lambda_kl=-1., CL_neg_samples=1):
    # two sides
    CL_loss_1, KL_loss_1 = do_CL(X, Y, normalize, metric, T, lambda_cl, lambda_kl, CL_neg_samples)
    CL_loss_2, KL_loss_2 = do_CL(Y, X, normalize, metric, T, lambda_cl, lambda_kl, CL_neg_samples)
    # return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2
    return (CL_loss_1 + CL_loss_2) / 2, (KL_loss_1 + KL_loss_2) / 2


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_file_logger(file_name: str = 'log.txt', log_format: str = '%(message)s', log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    handler = logging.FileHandler(file_name, "w")
    handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


if __name__ == '__main__':
    print()
