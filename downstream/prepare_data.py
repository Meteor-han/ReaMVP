from utils_ds import *
from utils import mol_to_dgl_graph
from data_utils.merge_data import split_line
from rdkit.Chem import AllChem
import pickle
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')
vocab_file = os.path.join("data", "pretraining_data", "merge_vocab_frequency.pt")
with open(vocab_file, "rb") as f:
    frequency_vocab = pickle.load(f)


# both graph and graph_radius
def get_dgl_with_dict(smi, smi_dgl_dict=None):
    if smi_dgl_dict is None:
        smi_dgl_dict = {}
    if smi in smi_dgl_dict:
        return smi_dgl_dict[smi]
    else:
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m)
        m = Chem.RemoveHs(m)
        # new_s = Chem.MolToSmiles(m)
        c = m.GetConformers()[0]
        p = c.GetPositions()
        s_dgl_2d = mol_to_dgl_graph(m, False)
        s_dgl_3d = mol_to_dgl_graph(m, True, p, radius=10.0)
        smi_dgl_dict[smi] = [s_dgl_2d, s_dgl_3d]
        return [s_dgl_2d, s_dgl_3d]


def compute_save_BH(dataset, save_dir):
    # already canonicalized
    # BH, 01, 2345, 6
    BH_list = []
    # l, c, r
    smiles_index_dict = {}
    # s
    smiles_dgl_dict = {}
    # does not contain single one
    for i in range(len(dataset)):
        temp_reaction = []
        one_dgl_list = []
        for s in dataset[i][:-1]:
            # mol = Chem.MolFromSmiles(s)
            # if mol.GetNumAtoms() < 1.5:
            #     exit()
            temp_reaction.append(s)
            one_dgl_list.append(get_dgl_with_dict(s, smiles_dgl_dict))
        l = ".".join(temp_reaction[:2])
        c = ".".join(temp_reaction[2:6])
        r = temp_reaction[6]
        for s in [l, c, r]:
            temp_index_list = []
            for one in split_line(s):
                temp_index_list.append(frequency_vocab.get(one, frequency_vocab["UNK"]))
            smiles_index_dict[s] = temp_index_list
        one_index_list = [frequency_vocab["BOS"]] + smiles_index_dict[l] + [frequency_vocab["SEP"]] + \
                         smiles_index_dict[c] + [frequency_vocab["SEP"]] + smiles_index_dict[r]
        one_index_list = one_index_list[:511] + [frequency_vocab["EOS"]]
        BH_list.append([dataset[i], one_index_list, one_dgl_list, dataset[i][-1]])
    with open(save_dir, "wb") as f:
        pickle.dump([BH_list, smiles_index_dict, smiles_dgl_dict], f)
    return BH_list, smiles_index_dict, smiles_dgl_dict


def compute_save_SM(dataset, save_dir, tag=False):
    # already canonicalized
    # SM, ~, -1
    SM_list = []
    # l, c, r
    smiles_index_dict = {}
    # s
    smiles_dgl_dict = {}
    for i in range(len(dataset)):
        temp_reaction = []
        one_dgl_list = []
        char_ = "?" if tag else "."
        l, c, r = dataset[i][0].split(">")
        l, c, r = l.replace("?", "."), c.replace("?", "."), r.replace("?", ".")
        for s in dataset[i][0].replace("?", char_).replace(">", char_).split(char_):
            # omit single one?
            # mol = Chem.MolFromSmiles(s)
            # if mol.GetNumAtoms() < 1.5:
            #     continue
            temp_reaction.append(s)
            one_dgl_list.append(get_dgl_with_dict(s, smiles_dgl_dict))
        for s in [l, c, r]:
            temp_index_list = []
            for one in split_line(s):
                temp_index_list.append(frequency_vocab.get(one, frequency_vocab["UNK"]))
            smiles_index_dict[s] = temp_index_list
        one_index_list = [frequency_vocab["BOS"]] + smiles_index_dict[l] + \
                         [frequency_vocab["SEP"]] + smiles_index_dict[c] + \
                         [frequency_vocab["SEP"]] + smiles_index_dict[r]
        one_index_list = one_index_list[:511] + [frequency_vocab["EOS"]]
        SM_list.append([dataset[i], one_index_list, one_dgl_list, dataset[i][-1]])
    with open(save_dir, "wb") as f:
        pickle.dump([SM_list, smiles_index_dict, smiles_dgl_dict], f)
    return SM_list, smiles_index_dict, smiles_dgl_dict


if __name__ == '__main__':
    df_BH = pd.read_csv(os.path.join("data", "BH/BH.csv"), sep=',')
    dataset_BH = generate_buchwald_hartwig_rxns(df_BH, 0.01)
    BH_ = compute_save_BH(dataset_BH, os.path.join("data", "BH/BH_index_dgl_dict.pt"))

    df_SM = pd.read_csv(os.path.join("data", "SM/SM_custom.tsv"), sep='\t')
    dataset_SM = generate_s_m_rxns(df_SM, 0.01)
    SM = compute_save_SM(dataset_SM, os.path.join("data", "SM/SM_index_dgl_dict.pt"))
    print()
