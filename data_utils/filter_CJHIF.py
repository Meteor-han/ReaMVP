import os
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from collections import defaultdict
import time
import pickle
from urllib.request import urlopen
from urllib.parse import quote
# import cirpy
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def cir_convert(ids):
    # the website sometimes "died"
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return ""


def cata_name_to_smiles_opsin(cjhif_file=os.path.join("..", "data", "original_data", "data_from_CJHIF_utf8"),
                              input_file=os.path.join("..", "data", "pretraining_data", "input.txt"),
                              output_file=os.path.join("..", "data", "pretraining_data", "output.txt")):
    with open(cjhif_file, "r", encoding="utf8") as f:
        total = f.readlines()
    cata = set()
    for i in range(len(total)):
        temp = total[i].split("§")[3:5]
        for one in temp:
            if one and one not in cata:
                cata.add(one)
    cata_single = []
    for one in cata:
        # ","
        temp = one.replace("\\", "").replace(";", "|")
        temp = temp.split("|")
        cata_single.extend(temp)
    cata_single = list(set(cata_single))

    print("Reagent num: {}".format(len(cata_single)))
    with open(input_file, "w") as f:
        for one in cata_single:
            f.write(one + "\n")
    command = "java -jar opsin-cli-2.7.0-jar-with-dependencies.jar -osmi {} {}".format(input_file, output_file)
    os.system(command)


def cata_name_to_smiles_cir(input_file=os.path.join("..", "data", "pretraining_data", "input.txt"),
                            output_file=os.path.join("..", "data", "pretraining_data", "output_CIR.txt")):
    with open(input_file, "r") as f:
        cata_name = f.readlines()
    smiles = []
    start = time.time()
    for i in range(len(cata_name)):
        if (i + 1) % 10 == 0:
            print(i, time.time() - start)
        c = cata_name[i].strip()
        # cirpy.resolve(c, 'smiles'), CIRconvert(c)
        smiles.append(cir_convert(c))
    with open(output_file, "w") as f:
        for one in smiles:
            f.write(one + "\n")


def trans(smiles):
    # isomericSmiles, kekuleSmiles (F), canonical
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def filter_duplicate_invalid(cjhif_file=os.path.join("..", "data", "original_data", "data_from_CJHIF_utf8"),
                             cjhif_file_new=os.path.join("..", "data", "pretraining_data", "CJHIF_valid.pt"),
                             input_file=os.path.join("..", "data", "pretraining_data", "input.txt"),
                             output_opsin=os.path.join("..", "data", "pretraining_data", "output.txt")):
    # output_cir=os.path.join("..", "data", "pretraining_data", "output_CIR.txt")
    with open(input_file, "r") as f:
        cata_name = f.readlines()
    with open(output_opsin, "r") as f:
        cata_smiles_opsin = f.readlines()
    # with open(output_cir, "r") as f:
    #     cata_smiles_cir = f.readlines()
    name_smiles = {}
    for i in range(len(cata_name)):
        s_opsin = cata_smiles_opsin[i].strip()
        # s_cir = cata_smiles_cir[i].strip()
        c = cata_name[i].strip()
        if s_opsin:
            try:
                name_smiles[c] = trans(s_opsin)
            except:
                pass
        # elif s_cir:
        #     try:
        #         name_smiles[c] = trans(s_cir)
        #     except:
        #         pass

    with open(cjhif_file, "r", encoding="utf8") as f:
        total = f.readlines()
    print("Original num: {}".format(len(total)))
    start = time.time()
    filtered_reactions = defaultdict(list)
    cat_set = []
    cat_set_single_n = []
    for i in range(len(total)):
        if (i + 1) % 50000 == 0:
            print("{}, time: {}".format(i + 1, time.time() - start))
        temp = total[i].split("§")
        # all reagents, emmm maybe duplicate here
        temp_cata_n = temp[3] + "|" + temp[4] + "|" + temp[5]
        temp_cata_n = temp_cata_n.replace("\\", "").replace(";", "|")
        temp_cata_n = temp_cata_n.split("|")
        temp_cata_s = ""
        cat_set_single_n.extend(temp_cata_n)
        temp_cata_n = set(temp_cata_n)
        for one in temp_cata_n:
            if one in name_smiles:
                temp_cata_s += (name_smiles[one] + ".")
        # {""}[-1] = ""
        cat_set.append(temp_cata_s[:-1])
        smarts = temp[2].replace(";", ".") + temp[0].split(">")[0] + ">" \
                 + temp_cata_s[:-1] + ">" + temp[0].split(">")[2]
        temp_y = float(temp[6])
        try:
            # SMILES input is OK. It will order the string.
            # e.g. CN.NC=O.O=C1CCCN1C1CCN(Cc2ccccc2)CC1 whatever the input order of these three molecules
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
            filtered_reactions[(".".join(l), ".".join(c), tuple(temp_cata_n), ".".join(r))].append(temp_y)
        except:
            pass
    end = time.time() - start
    with open(cjhif_file_new, "wb") as f:
        pickle.dump(filtered_reactions, f)
    cat_set_single_n = set(cat_set_single_n)
    cat_set = set(cat_set)
    print("Filtered num: {} time: {}\n"
          "Cata single num: {}\n"
          "Cata reaction num: {}".format(len(filtered_reactions), end, len(cat_set_single_n), len(cat_set)))


def extract_yield(cjhif_file=os.path.join("..", "data", "pretraining_data", "CJHIF_valid.pt"),
                  cjhif_file_with_yield=os.path.join("..", "data", "pretraining_data", "CJHIF_valid_with_yield.pt")):
    # (l, c, name, r): [y, ...]
    with open(cjhif_file, "rb") as f:
        reactions = pickle.load(f)
    print("Original num: {}".format(len(reactions)))

    reactions_dict = defaultdict(list)
    for (l, c, name, r), v in reactions.items():
        reactions_dict[(l, c, r)] = v

    reactions_dict_yield = defaultdict(float)
    for (l, c, r), v in reactions_dict.items():
        if len(v) != 1:
            if ((max(v) - min(v)) < 20) and (max(v) <= 100) and (min(v) >= 0):
                final_y = sum(v) / len(v)
            else:
                continue
        else:
            final_y = v[0]
        reactions_dict_yield[(l, c, r)] = final_y
    with open(cjhif_file_with_yield, "wb") as f:
        pickle.dump(reactions_dict_yield, f)
    print("Filtered num: {}".format(len(reactions_dict_yield)))


def select_by_hand(cjhif_file=os.path.join("..", "data", "pretraining_data", "CJHIF_valid_with_yield.pt"),
                   cjhif_file_sub=os.path.join("..", "data", "pretraining_data", "CJHIF_valid_with_yield_sub.pt")):
    with open(cjhif_file, "rb") as f:
        reactions = pickle.load(f)
    print("Original num: {}".format(len(reactions)))

    reactions_sub = {}
    num_10 = 0
    num_50 = 0
    reactions_list = []
    for (l, c, r), v in reactions.items():
        reactions_list.append([l, c, r, v])
    import random
    random.shuffle(reactions_list)
    for l, c, r, v in reactions_list:
        if v <= 10:
            if num_10 < 26000:
                reactions_sub[(l, c, r)] = v
                num_10 += 1
        elif v <= 50:
            if num_50 < 100000:
                r_seed = random.uniform(15, 51)
                if r_seed >= v:
                    reactions_sub[(l, c, r)] = v
                    num_50 += 1
    with open(cjhif_file_sub, "wb") as f:
        pickle.dump(reactions_sub, f)
    print("Filtered num: {}".format(len(reactions_sub)))


if __name__ == '__main__':
    # cata_name_to_smiles_opsin()
    # cata_name_to_smiles_cir()
    # filter_duplicate_invalid()
    # extract_yield()
    # select_by_hand()
    print()
