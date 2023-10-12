import os
import time
import pickle
from utils import *
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# caution: Kekule form (no aromatic bonds), lower case, what about [cs], ccs?
# length of 2 only
Elements_2 = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al',
              'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
              'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc',
              'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
              'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
              'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
              'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Lr', 'Rf', 'Db', 'Sg',
              'Bh', 'Hs', 'Mt', 'Ds', 'Rg']
Elements_2_lower = ['he', 'li', 'be', 'ne', 'na', 'mg', 'al',
                    'si', 'cl', 'ar', 'ca', 'sc', 'ti', 'cr', 'mn', 'fe', 'co', 'ni',
                    'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'zr', 'nb', 'mo', 'tc',
                    'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'xe', 'cs', 'ba', 'la', 'ce',
                    'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta',
                    're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra',
                    'ac', 'th', 'pa', 'np', 'pu', 'am', 'cm', 'bk', 'cf', 'es', 'lr', 'rf', 'db', 'sg',
                    'bh', 'hs', 'mt', 'ds', 'rg']

# Zhao etc.
Elements = ['H', 'He', 'Li', 'Be', '', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
            'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
            'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
            'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
            'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
            'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Lr', 'Rf', 'Db', 'Sg',
            'Bh', 'Hs', 'Mt', 'Ds', 'Rg']


# []
# def tokenize(smi):
#     smi += " "
#     tokens = []
#     marker = 0  # 1 indicates that the last token is not complete
#     i = 0
#     while i <= len(smi) - 2:
#         temp = smi[i:i + 2]
#         c = smi[i]
#         if marker == 0 and temp in Elements_2:
#             tokens.append(temp)
#             i += 2
#         else:
#             if marker == 0:
#                 tokens.append(c)
#                 if c == '[':
#                     marker = 1
#                 elif c == '%':
#                     tokens[-1] += smi[i + 1: i + 3]
#                     i += 2
#             else:
#                 tokens[-1] += c
#                 if c == ']':
#                     marker = 0
#             i += 1
#     return tokens


# tokenizer
def split_line(input_str):
    tmp = ''
    input_str += ' '
    i = 0
    while i <= len(input_str) - 2:
        #        print i
        sub = input_str[i: i + 2]
        tmp_chr = input_str[i + 2] if i < len(input_str) - 2 else 'x'
        if sub in Elements_2 and not ('9' > tmp_chr > '0' or tmp_chr == '%' or tmp_chr == 'x'):
            tmp += (sub + ' ')
            i = i + 1
        elif input_str[i] == '%':
            tmp += (input_str[i: i + 3] + ' ')
            i = i + 2
        elif sub in Elements_2_lower and not ('9' > tmp_chr > '0' or tmp_chr == '%' or tmp_chr == 'x'):
            try:
                if input_str[i - 1] == "[" or input_str[i + 2] == "]":
                    tmp += (sub + ' ')
                    i = i + 1
                else:
                    tmp += (input_str[i] + ' ')
            except:
                tmp += (input_str[i] + ' ')
        else:
            tmp += (input_str[i] + ' ')
        i += 1
        # print(tmp)
    return tmp.strip().replace('  ', ' ').split()


def merge_data(uspto_dict=os.path.join("..", "data", "pretraining_data", "USPTO_valid.pt"),
               uspto_dict_with_yield=os.path.join("..", "data", "pretraining_data", "USPTO_valid_with_yield.pt"),
               cjhif_dict_with_yield=os.path.join("..", "data", "pretraining_data", "CJHIF_valid_with_yield_sub.pt"),
               cl_file=os.path.join("..", "data", "pretraining_data", "merge_cl.pt"),
               yield_file=os.path.join("..", "data", "pretraining_data", "merge_yield.pt")):
    with open(uspto_dict, "rb") as f:
        uspto_cl = pickle.load(f)
    with open(uspto_dict_with_yield, "rb") as f:
        uspto_yield = pickle.load(f)
    with open(cjhif_dict_with_yield, "rb") as f:
        cjhif_yield = pickle.load(f)

    merge_cl = list(uspto_cl.keys())

    merge_yield_dict = {}
    merge_yield = []

    duplicate_num = 0
    delete_num = 0
    uspto_k = set(list(uspto_yield.keys()))
    for (l, c, r), y in uspto_yield.items():
        merge_yield_dict[(l, c, r)] = y
    for (l, c, r), y in cjhif_yield.items():
        if (l, c, r) in uspto_k:
            duplicate_num += 1
            if abs(merge_yield_dict[(l, c, r)] - y) < 20:
                merge_yield_dict[(l, c, r)] = (merge_yield_dict[(l, c, r)] + y) / 2
            else:
                delete_num += 1
                merge_yield_dict.pop((l, c, r))
        else:
            merge_yield_dict[(l, c, r)] = y
    for (l, c, r), y in merge_yield_dict.items():
        merge_yield.append([l, c, r, y])

    with open(cl_file, "wb") as f:
        pickle.dump(merge_cl, f)
    with open(yield_file, "wb") as f:
        pickle.dump(merge_yield, f)
    print("Merge yield num: {}\n Duplicate num: {}\n Delete num: {}".
          format(len(merge_yield), duplicate_num, delete_num))


def get_vocab(cl_file=os.path.join("..", "data", "pretraining_data", "merge_cl.pt"),
              yield_file=os.path.join("..", "data", "pretraining_data", "merge_yield.pt"),
              split_file=os.path.join("..", "data", "pretraining_data", "merge_split_line_dict.pt"),
              vocab_file=os.path.join("..", "data", "pretraining_data", "merge_vocab.pt")
              ):
    with open(cl_file, "rb") as f:
        cl_data = pickle.load(f)
    with open(yield_file, "rb") as f:
        yield_data = pickle.load(f)
    smiles_set = set()
    for (l, c, r) in cl_data:
        smiles_set.add(l)
        smiles_set.add(c)
        smiles_set.add(r)
    for (l, c, r, y) in yield_data:
        smiles_set.add(l)
        smiles_set.add(c)
        smiles_set.add(r)
    # SMILES -> split string
    split_dict = dict()
    print("SMILES num: {}".format(len(smiles_set)))
    start_t = time.time()
    for i, one in enumerate(smiles_set):
        if (i + 1) % 1e5 == 0:
            print("{}, time: {}".format(i + 1, time.time() - start_t))
        split_dict[one] = split_line(one)
    base_vocab = {"UNK": 0, "BOS": 1, "EOS": 2, "SEP": 3, "PAD": 4}
    index = len(base_vocab.keys())
    for smiles, seq in split_dict.items():
        for ele in seq:
            if ele not in base_vocab.keys():
                base_vocab[ele] = index
                index += 1
    with open(vocab_file, "wb") as f:
        pickle.dump(base_vocab, f)
    with open(split_file, "wb") as f:
        pickle.dump(split_dict, f)
    print("Raw vocab size: {}".format(len(base_vocab)))


def check_vocab():
    vocab_file = os.path.join("..", "data", "pretraining_data", "merge_vocab.pt")
    with open(vocab_file, "rb") as f:
        base_vocab = pickle.load(f)

    # index -> ele 0->"UNK"
    reverse_vocab = {}
    for k, v in base_vocab.items():
        reverse_vocab[v] = k
    cl_file = os.path.join("..", "data", "pretraining_data", "merge_cl.pt")
    yield_file = os.path.join("..", "data", "pretraining_data", "merge_yield.pt")
    with open(cl_file, "rb") as f:
        cl_data = pickle.load(f)
    with open(yield_file, "rb") as f:
        yield_data = pickle.load(f)
    split_file = os.path.join("..", "data", "pretraining_data", "merge_split_line_dict.pt")
    with open(split_file, "rb") as f:
        split_dict = pickle.load(f)
    # SMILES -> index
    index_dict = dict()
    for smiles, seq in split_dict.items():
        temp_index = []
        for ele in seq:
            temp_index.append(base_vocab[ele])
        index_dict[smiles] = temp_index

    cl_index_data, yield_index_data = [], []
    for (l, c, r) in cl_data:
        cl_index_data.append(tuple([base_vocab["BOS"]] + index_dict[l] + [base_vocab["SEP"]] +
                                   index_dict[c] + [base_vocab["SEP"]] + index_dict[r] + [base_vocab["EOS"]]))
    for (l, c, r, y) in yield_data:
        yield_index_data.append(tuple([base_vocab["BOS"]] + index_dict[l] + [base_vocab["SEP"]] +
                                      index_dict[c] + [base_vocab["SEP"]] + index_dict[r] + [base_vocab["EOS"]]))
    yield_index_data = []
    total_index = set(cl_index_data).union(set(yield_index_data))
    # ele -> frequency
    from collections import defaultdict
    dict_count = defaultdict(int)
    for i, seq in enumerate(total_index):
        # if i % 1e4 == 0:
        #     print(i)
        for ele_index in seq:
            dict_count[reverse_vocab[ele_index]] += 1
    fre = sorted(dict_count.items(), key=lambda x: x[1], reverse=False)
    # check the above frequency!
    fre_file = os.path.join("..", "data", "pretraining_data", "merge_vocab_frequency_raw.pt")
    with open(fre_file, "wb") as f:
        pickle.dump(fre, f)
    print(fre)

    unk = []
    for one in fre:
        if one[1] >= 10:
            break
        if base_vocab[one[0]] >= 5:
            unk.append(one[0])
    frequency_vocab = {}
    for ele, ele_index in base_vocab.items():
        if ele in unk:
            frequency_vocab[ele] = base_vocab["UNK"]
        else:
            frequency_vocab[ele] = ele_index

    vocab_file_fre = os.path.join("..", "data", "pretraining_data", "merge_vocab_frequency.pt")
    with open(vocab_file_fre, "wb") as f:
        pickle.dump(frequency_vocab, f)


# check pos, if memory ok, dgl graph
def check_memory(cl_file=os.path.join("..", "data", "pretraining_data", "merge_cl.pt"),
                 yield_file=os.path.join("..", "data", "pretraining_data", "merge_yield.pt"),
                 split_file=os.path.join("..", "data", "pretraining_data", "merge_split_line_dict.pt"),
                 vocab_file=os.path.join("..", "data", "pretraining_data", "merge_vocab_frequency.pt"),
                 pos_file=os.path.join("..", "data", "pretraining_data", "smiles_pos_dict.pt"),
                 cl_dgl_file=os.path.join("..", "data", "pretraining_data", "merge_cl_dgl.pt"),
                 yield_dgl_file=os.path.join("..", "data", "pretraining_data", "merge_yield_dgl.pt")):
    # remember, max len undo
    with open(cl_file, "rb") as f:
        cl_data = pickle.load(f)
    with open(yield_file, "rb") as f:
        yield_data = pickle.load(f)

    with open(vocab_file, "rb") as f:
        frequency_vocab = pickle.load(f)
    with open(split_file, "rb") as f:
        split_dict = pickle.load(f)

    with open(pos_file, "rb") as f:
        smiles_pos_dict = pickle.load(f)

    # SMILES -> index
    index_dict = dict()
    for smiles, seq in split_dict.items():
        temp_index = []
        for ele in seq:
            temp_index.append(frequency_vocab[ele])
        index_dict[smiles] = temp_index

    # about 500Mb for 10000 reactions, unacceptable!
    # dgl without pos? then yield load?
    # [(l, c, r), index list for RNN, dgl list, yield if given]
    cl_dgl_data, yield_dgl_data = [], []
    cl_num_fail, yield_num_fail = 0, 0
    cl_smiles_fail, yield_smiles_fail = set(), set()
    start_t = time.time()
    for idd, (l, c, r) in enumerate(cl_data):
        if (idd + 1) >= 100001:
            break
        if (idd + 1) % 50000 == 0:
            print("{}, {}, time: {}".format(idd + 1, cl_num_fail, time.time() - start_t))
        # [...]+[""]+[...], stupid man!
        reaction = l.split(".") + c.split(".") + r.split(".") if c else l.split(".") + r.split(".")
        one_dgl_gs = []
        flag = False
        for s in reaction:
            if s not in smiles_pos_dict:
                cl_smiles_fail.add(s)
                cl_num_fail += 1
                flag = True
                break
            m = Chem.MolFromSmiles(s)
            temp_dgl = mol_to_dgl_graph(m, use_3d=True, pos=smiles_pos_dict[s], radius=10.0)
            if temp_dgl is None:
                cl_smiles_fail.add(s)
                cl_num_fail += 1
                flag = True
                break
            one_dgl_gs.append(temp_dgl)
        if flag:
            continue
        one_index_list = [frequency_vocab["BOS"]] + index_dict[l] + [frequency_vocab["SEP"]] + \
                         index_dict[c] + [frequency_vocab["SEP"]] + index_dict[r] + [frequency_vocab["EOS"]]
        cl_dgl_data.append([(l, c, r), one_index_list, one_dgl_gs])
    with open(cl_dgl_file, "wb") as f:
        pickle.dump(cl_dgl_data, f)
    print("Cl fail num: {}\n"
          "Cl final num: {}".format(cl_num_fail, len(cl_dgl_data)))
    # print(len(cl_smiles_fail))
    # print(cl_smiles_fail)
    del cl_dgl_data
    exit()

    for idd, (l, c, r, y) in enumerate(yield_data):
        if (idd + 1) % 50000 == 0:
            print("{}, {}, time: {}".format(idd + 1, yield_num_fail, time.time() - start_t))
        reaction = l.split(".") + c.split(".") + r.split(".")
        one_dgl_gs = []
        flag = False
        for s in reaction:
            if s:
                if s not in smiles_pos_dict:
                    yield_smiles_fail.add(s)
                    yield_num_fail += 1
                    flag = True
                    break
                m = Chem.MolFromSmiles(s)
                temp_dgl = mol_to_dgl_graph(m, use_3d=True, pos=smiles_pos_dict[s], radius=10.0)
                if temp_dgl is None:
                    cl_smiles_fail.add(s)
                    cl_num_fail += 1
                    flag = True
                    break
                one_dgl_gs.append(temp_dgl)
        if flag:
            continue
        one_index_list = [frequency_vocab["BOS"]] + index_dict[l] + [frequency_vocab["SEP"]] + \
                         index_dict[c] + [frequency_vocab["SEP"]] + index_dict[r] + [frequency_vocab["EOS"]]
        yield_dgl_data.append([(l, c, r), one_index_list, one_dgl_gs, y])
    with open(yield_dgl_file, "wb") as f:
        pickle.dump(yield_dgl_data, f)
    print("Yield fail num: {}\n"
          "Yield final num: {}".format(yield_num_fail, len(yield_dgl_data)))
    del yield_dgl_data


def get_dgl_dict(pos_file=os.path.join("..", "data", "pretraining_data", "smiles_pos_dict.pt"),
                 dgl_file=os.path.join("..", "data", "pretraining_data", "smiles_dgl_dict.pt")):
    with open(pos_file, "rb") as f:
        smiles_pos_dict = pickle.load(f)
    smiles_dgl_dict = {}
    start_t = time.time()
    for i, (k, v) in enumerate(smiles_pos_dict.items()):
        if (i + 1) % 20000 == 0:
            print("{}, time: {}".format(i + 1, time.time() - start_t))
        m = Chem.MolFromSmiles(k)
        temp_dgl = mol_to_dgl_graph(m, use_3d=True, pos=smiles_pos_dict[k], radius=10.0)
        if temp_dgl is not None:
            smiles_dgl_dict[k] = temp_dgl
    with open(dgl_file, "wb") as f:
        pickle.dump(smiles_dgl_dict, f)
    with open(dgl_file[:-3] + "_smiles.pt", "wb") as f:
        pickle.dump(set(smiles_dgl_dict.keys()), f)
    print(len(smiles_dgl_dict))


def prepare_pretraining_data(
        cl_file=os.path.join("..", "data", "pretraining_data", "merge_cl.pt"),
        yield_file=os.path.join("..", "data", "pretraining_data", "merge_yield.pt"),
        split_file=os.path.join("..", "data", "pretraining_data", "merge_split_line_dict.pt"),
        vocab_file=os.path.join("..", "data", "pretraining_data", "merge_vocab_frequency.pt"),
        smiles_file=os.path.join("..", "data", "pretraining_data", "smiles_valid.pt"),
        index_file=os.path.join("..", "data", "pretraining_data", "smiles_lcr_index_dict.pt"),
        pretraining_cl_file=os.path.join("..", "data", "pretraining_data", "pretraining_cl.pt"),
        pretraining_yield_file=os.path.join("..", "data", "pretraining_data", "pretraining_yield.pt")):
    # remember, max len undo
    with open(cl_file, "rb") as f:
        cl_data = pickle.load(f)
    with open(yield_file, "rb") as f:
        yield_data = pickle.load(f)

    with open(vocab_file, "rb") as f:
        frequency_vocab = pickle.load(f)
    with open(split_file, "rb") as f:
        split_dict = pickle.load(f)

    with open(smiles_file, "rb") as f:
        smiles_set = pickle.load(f)

    # SMILES -> index
    index_dict = dict()
    for smiles, seq in split_dict.items():
        temp_index = []
        for ele in seq:
            temp_index.append(frequency_vocab[ele])
        index_dict[smiles] = temp_index
    with open(index_file, "wb") as f:
        pickle.dump(index_dict, f)

    pretraining_cl_data, pretraining_yield_data = [], []
    cl_num_fail, yield_num_fail = 0, 0
    cl_smiles_fail, yield_smiles_fail = set(), set()
    start_t = time.time()
    for idd, (l, c, r) in enumerate(cl_data):
        if (idd + 1) % 50000 == 0:
            print("{}, {}, time: {}".format(idd + 1, cl_num_fail, time.time() - start_t))
        # [...]+[""]+[...], stupid man!
        reaction = l.split(".") + c.split(".") + r.split(".") if c else l.split(".") + r.split(".")
        flag = False
        for s in reaction:
            if s and (s not in smiles_set):
                cl_smiles_fail.add(s)
                cl_num_fail += 1
                flag = True
                break
        if flag:
            continue
        one_index_list = [frequency_vocab["BOS"]] + index_dict[l] + [frequency_vocab["SEP"]] + \
                         index_dict[c] + [frequency_vocab["SEP"]] + index_dict[r]
        one_index_list = one_index_list[:511] + [frequency_vocab["EOS"]]
        pretraining_cl_data.append((l, c, r, reaction, one_index_list))
    print("Cl fail num: {}\n"
          "Cl final num: {}".format(cl_num_fail, len(pretraining_cl_data)))
    # print(len(cl_smiles_fail))
    # print(cl_smiles_fail)

    for idd, (l, c, r, y) in enumerate(yield_data):
        if (idd + 1) % 50000 == 0:
            print("{}, {}, time: {}".format(idd + 1, yield_num_fail, time.time() - start_t))
        reaction = l.split(".") + c.split(".") + r.split(".")
        flag = False
        for s in reaction:
            if s and (s not in smiles_set):
                yield_smiles_fail.add(s)
                yield_num_fail += 1
                flag = True
                break
        if flag:
            continue
        one_index_list = [frequency_vocab["BOS"]] + index_dict[l] + [frequency_vocab["SEP"]] + \
                         index_dict[c] + [frequency_vocab["SEP"]] + index_dict[r]
        one_index_list = one_index_list[:511] + [frequency_vocab["EOS"]]
        pretraining_yield_data.append((l, c, r, reaction, one_index_list, y))
    print("Yield fail num: {}\n"
          "Yield final num: {}".format(yield_num_fail, len(pretraining_yield_data)))

    # training, validation, test, 0.9, 0.05, 0.05
    num_cl = len(pretraining_cl_data)
    import random
    random.shuffle(pretraining_cl_data)
    for (name, start_id, end_id) in [("training", 0, int(num_cl * 0.9)),
                                     ("validation", int(num_cl * 0.9), int(num_cl * 0.95)),
                                     ("test", int(num_cl * 0.95), num_cl)]:
        with open(pretraining_cl_file[:-3] + "_{}.pt".format(name), "wb") as f:
            pickle.dump(pretraining_cl_data[start_id:end_id], f)
        print(len(pretraining_cl_data[start_id:end_id]))

    ys = []
    for (l, c, r, reaction, one_index_list, y) in pretraining_yield_data:
        ys.append(float(y))
    # stratified sampling, 10
    num_level = 10
    range_level = 100 / num_level
    index_list = []
    for i in range(num_level):
        index_list.append([])
    for i, one_y in enumerate(ys):
        index_list[min(int(one_y / range_level), num_level - 1)].append(i)
    # split 9:0.5:0.5
    split = [0.9, 0.05]
    temp_index, test_index = [], []
    rate = split[0] + split[1]
    for one in index_list:
        sampled_index = random.sample(one, k=int(len(one) * rate))
        temp_index.append(sampled_index)
        test_index.extend(list(set(one).difference(set(sampled_index))))
    training_index, val_index = [], []
    rate = split[0] / rate
    for one in temp_index:
        sampled_index = random.sample(one, k=int(len(one) * rate))
        training_index.extend(sampled_index)
        val_index.extend(list(set(one).difference(set(sampled_index))))
    training_data, val_data, test_data = [], [], []
    for (one_index, one_data) in zip([training_index, val_index, test_index], [training_data, val_data, test_data]):
        for one in one_index:
            one_data.append(pretraining_yield_data[one])
    random.shuffle(training_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    for (name, dataset) in [("training", training_data),
                            ("validation", val_data),
                            ("test", test_data)]:
        with open(pretraining_yield_file[:-3] + "_{}.pt".format(name), "wb") as f:
            pickle.dump(dataset, f)
        print(len(dataset))


if __name__ == '__main__':
    # merge_data()
    # get_vocab()
    # check_vocab()
    # check_memory()
    # get_dgl_dict()
    # prepare_pretraining_data()
    print()
