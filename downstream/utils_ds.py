import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions


def trans(smiles):
    # isomericSmiles, kekuleSmiles (F), canonical
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def canonicalize_with_dict(smi, can_smi_dict=None):
    if can_smi_dict is None:
        can_smi_dict = {}
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]


# return: a list of tuples, [(SMILES_1, ..., SMILES_n, Yield), ...]
def generate_buchwald_hartwig_rxns(df, mul=1.0):
    """
    Converts the entries in the excel files from Sandfort et al. to reaction SMILES.
    """
    df = df.copy()
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>' \
                   '[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
    methylaniline = 'Cc1ccc(N)cc1'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []
    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row['Aryl halide']), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])
    df['product'] = products
    rxns = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive

        reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        rxns.append(tuple(f"{reactants}.{row['product']}".split(".")) + (row['Output'] * mul,))
    return rxns


def generate_s_m_rxns(df, mul=1.0):
    rxns = []
    for i, row in df.iterrows():
        rxns.append((row['rxn'],) + (row['y'] * 100 * mul,))  # .replace(">>", ".").split(".")
    return rxns


def generate_buchwald_hartwig_wo_additive(df, mul=0.01):
    df = df.copy()
    methylaniline = 'Cc1ccc(N)cc1'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))

    rxns = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        try:
            aryl_halide = canonicalize_with_dict(row['aryl_halide_smiles'], can_smiles_dict)
            can_smiles_dict[row['aryl_halide_smiles']] = aryl_halide
            ligand = canonicalize_with_dict(row['ligand_smiles'], can_smiles_dict)
            can_smiles_dict[row['ligand_smiles']] = ligand
            base = canonicalize_with_dict(row['base_smiles'], can_smiles_dict)
            can_smiles_dict[row['base_smiles']] = base
            product = canonicalize_with_dict(row['product_smiles'], can_smiles_dict)
            can_smiles_dict[row['product_smiles']] = product

            reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{product}"
            rxns.append(tuple(f"{reactants}".split(".")) + (row['yield'] * mul,))
        except:
            continue
    return rxns


if __name__ == '__main__':
    df_BH = pd.read_csv("../../data/BH/BH.csv", sep=',')
    dataset_BH = generate_buchwald_hartwig_rxns(df_BH)
    df_SM = pd.read_csv("../../data/SM/SM.tsv", sep='\t')
    dataset_SM = generate_s_m_rxns(df_SM)
    df_BH_wo = pd.read_csv("../../data/BH_wo/data_table.csv", sep=',')
    dataset_BH_wo = generate_buchwald_hartwig_wo_additive(df_BH_wo)
    print()

    # import pickle
    # with open("../../data/AAA/AAA_3d.pt", "rb") as f:
    #     ds_dataset = pickle.load(f)
    print()
