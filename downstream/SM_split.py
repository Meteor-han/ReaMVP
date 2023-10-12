from collections import defaultdict
from utils_ds import *


def make_reaction_smiles(row, dict_):
    # dict_ already trans
    char_ = "."
    wo_split = ['6-chloroquinoline', '6-Bromoquinoline', '6-triflatequinoline', '6-Iodoquinoline']
    reactant_1 = dict_[row['Reactant_1_Name']].replace(".", char_) if row['Reactant_1_Name'] in wo_split else dict_[row['Reactant_1_Name']]
    precursors = f"{reactant_1}{char_}{dict_[row['Reactant_2_Name']]}>{dict_[row['Catalyst_1_Short_Hand']]}{char_}{dict_[row['Ligand_Short_Hand']]}{char_}{dict_[row['Reagent_1_Short_Hand']]}{char_}{dict_[row['Solvent_1_Short_Hand']]}"
    product = 'C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12'
    can_product = trans(product)
    # only ligand and reagent contain "None"
    return f"{precursors}>{can_product}".replace(char_+char_+char_, char_).replace(char_+char_, char_).replace(' '+char_, '').replace(char_+' ', '').replace(' ', '')


if __name__ == '__main__':
    df = pd.read_excel(os.path.join('data', 'SM', 'aap9112_data_file_s1.xlsx')).fillna("None")
    # isomericSmiles or not; omit H2O or not; whole molecule or split by '.'
    name_to_smiles = {
        '6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
        '6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
        '6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC',
        '6-Iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
        '6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl',  # '.O'
        'Potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+]',  # '.O'
        '6-Quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3',  # '.O'
        '2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O', 
        '2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4', 
        '2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
        '2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br',
        'Pd(OAc)2': 'CC(=O)O~CC(=O)O~[Pd]',
        'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C', 
        'P(Ph)3 ': 'c3c(P(c1ccccc1)c2ccccc2)cccc3', 
        'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C', 
        'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3', 
        'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
        'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5', 
        'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC', 
        'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe+2]', 
        'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4', 
        'dppf': '[CH]1C=CC=C1P(c1ccccc1)c1ccccc1.[CH]1C=CC=C1P(c1ccccc1)c1ccccc1.[Fe+2]', 
        'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C',
        'None': '',
        'NaOH': '[OH-].[Na+]', 
        'NaHCO3': '[Na+].OC([O-])=O', 
        'CsF': '[F-].[Cs+]', 
        'K3PO4': '[K+].[K+].[K+].[O-]P([O-])([O-])=O', 
        'KOH': '[K+].[OH-]', 
        'LiOtBu': '[Li+].[O-]C(C)(C)C', 
        'Et3N': 'CCN(CC)CC',
        'MeCN': 'CC#N',  # '.O'
        'THF': 'C1CCOC1',  # '.O'
        'DMF': 'CN(C)C=O',  # '.O'
        'MeOH': 'CO',  # '.O'
        'MeOH/H2O_V2 9:1': 'CO',  # '.O'
        'THF_V2': 'C1CCOC1'  # '.O'
    }
    ligand_smiles = {
        'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C', 
        'P(Ph)3 ': 'c3c(P(c1ccccc1)c2ccccc2)cccc3', 
        'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C', 
        'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3', 
        'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
        'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5', 
        'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC', 
        'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]', 
        'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4', 
        'dppf': 'C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]', 
        'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C',
        'None': ''
    }
    for k in name_to_smiles.keys():
        name_to_smiles[k] = trans(name_to_smiles[k])
    ligand_count = defaultdict(int)
    for i, row in df.iterrows():
        ligand_count[row['Ligand_Short_Hand']] += 1

    """get random split, the same index as previous works, 
    we only change the form xx>>xx to reactants>catalyst.ligand.reagent.solvent>product"""
    df['rxn'] = [make_reaction_smiles(row, name_to_smiles) for i, row in df.iterrows()]
    df['y'] = df['Product_Yield_PCT_Area_UV'] / 100.
    reactions_df = df[['rxn', 'y']]
    for seed in range(10):
        new_df = pd.read_csv(os.path.join('data', 'SM/random_split_{}.tsv'.format(seed)), sep='\t')
        new_df.rename(columns={"Unnamed: 0": "index"}, inplace=True)
        for i in range(5760):
            new_df.iloc[i, 1] = reactions_df.iloc[new_df.iloc[i, 0], 0]
        new_df.to_csv(os.path.join('data', 'SM/random_split_{}_custom.tsv'.format(seed)), sep='\t')
    # the same as split 0, just for training
    new_df = pd.read_csv(os.path.join('data', 'SM/SM.tsv'), sep='\t')
    new_df.rename(columns={"Unnamed: 0": "index"}, inplace=True)
    for i in range(5760):
        new_df.iloc[i, 1] = reactions_df.iloc[new_df.iloc[i, 0], 0]
    new_df.to_csv(os.path.join('data', 'SM/SM_custom.tsv'), sep='\t')

    """get ligand-based split, row xlsx, then own data type"""
    test_ = []
    for names in [["AmPhos", "CataCXium A", "Xantphos"], ["P(Ph)3", "P(Cy)3", "P(o-Tol)3"],
                  ["P(tBu)3", "dtbpf", "dppf"], ["None", "SPhos", "XPhos"]]:
        training_index, test_index = [], []
        for i, row in df.iterrows():
            if row['Ligand_Short_Hand'] in names:
                test_index.append(i)
            else:
                training_index.append(i)
        test_.append(df.reindex(training_index+test_index))
    # create an excel writer object
    with pd.ExcelWriter(os.path.join('data', 'SM', 'SM_Test.xlsx')) as writer:
        for i, new_df in enumerate(test_):
            new_df.to_excel(writer, sheet_name="Test_{}".format(i+1), index=False)
    print()
    for id_ in range(1, 5):
        df = pd.read_excel(os.path.join('data', 'SM', 'SM_Test.xlsx'), sheet_name="Test_{}".format(id_)).fillna("None")
        df["index"] = df["Reaction_No"] - 1
        df['rxn'] = [make_reaction_smiles(row, name_to_smiles) for i, row in df.iterrows()]
        df['y'] = df['Product_Yield_PCT_Area_UV'] / 100.
        reactions_df = df[['index', 'rxn', 'y']]
        reactions_df.to_csv(os.path.join('data', 'SM', 'SM_Test_{}.tsv'.format(id_)), sep='\t')
        print()
    print()
