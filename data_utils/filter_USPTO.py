import os
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import time
import pickle
from collections import defaultdict
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def filter_duplicate_invalid(uspto_file=os.path.join("..", "data", "original_data", "data_from_USPTO_utf8"),
                             uspto_file_new=os.path.join("..", "data", "pretraining_data", "USPTO_valid.pt")):
    filtered_reactions = defaultdict(list)
    with open(uspto_file, "r", encoding="utf8") as f:
        # ['ReactionSmiles', 'PatentNumber', 'ParagraphNum', 'Year', 'TextMinedYield', 'CalculatedYield']
        name_space = f.readline().strip("\n").split("\t")
        lines = f.readlines()
        print("Original num: {}".format(len(lines)))
        start = time.time()
        for i, line in enumerate(lines):
            if (i+1) % 50000 == 0:
                print("{}, time: {}".format(i+1, time.time() - start))
            line = line.strip("\n").split("\t")
            smarts = line[0].split()[0]
            # smarts = line[0] not enough
            yield_text = line[4].strip("%")
            yield_cal = line[5].strip("%")
            try:
                reaction = rdChemReactions.ReactionFromSmarts(smarts)
                rdChemReactions.RemoveMappingNumbersFromReactions(reaction)
                reaction_smiles = rdChemReactions.ReactionToSmiles(reaction)
                rea = reaction_smiles.split(">")
                # Not enough! For each SMILES!
                l, c, r = rea[0].split("."), rea[1].split("."), rea[2].split(".")
                for j in range(len(l)):
                    l[j] = trans(l[j])
                for j in range(len(r)):
                    r[j] = trans(r[j])
                for j in range(len(c)):
                    if c[j]:
                        c[j] = trans(c[j])
                filtered_reactions[(".".join(l), ".".join(c), ".".join(r))].append([yield_text, yield_cal])
            except:
                pass
    end = time.time() - start
    with open(uspto_file_new, "wb") as f:
        pickle.dump(filtered_reactions, f)
    print("Filtered num: {} time: {}".format(len(filtered_reactions), end))


def extract_yield(uspto_file=os.path.join("..", "data", "pretraining_data", "USPTO_valid.pt"),
                  uspto_file_new=os.path.join("..", "data", "pretraining_data", "USPTO_valid_with_yield.pt")):
    # (l, c, r): [[yield_text, yield_cal], ...]
    with open(uspto_file, "rb") as f:
        reactions_dict = pickle.load(f)
    print("Original num: {}".format(len(reactions_dict)))

    reactions_dict_yield = defaultdict(float)
    count_dict = defaultdict(int)
    for (l, c, r), v in reactions_dict.items():
        for yield_text, yield_cal in v:
            try:
                temp = float(yield_text)
                if 0 <= temp <= 100:
                    current_y = reactions_dict_yield[(l, c, r)]
                    reactions_dict_yield[(l, c, r)] = current_y + (temp - current_y)/(count_dict[(l, c, r)] + 1)
                    count_dict[(l, c, r)] += 1
            except:
                try:
                    temp = float(yield_cal)
                    if 0 <= temp <= 100:
                        current_y = reactions_dict_yield[(l, c, r)]
                        reactions_dict_yield[(l, c, r)] = current_y + (temp - current_y) / (count_dict[(l, c, r)] + 1)
                        count_dict[(l, c, r)] += 1
                except:
                    continue
    with open(uspto_file_new, "wb") as f:
        pickle.dump(reactions_dict_yield, f)
    print("Filtered num: {}".format(len(reactions_dict_yield)))


def trans(smiles):
    # isomericSmiles, kekuleSmiles (F), canonical
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


if __name__ == '__main__':
    # filter_duplicate_invalid()
    # extract_yield()
    print()
