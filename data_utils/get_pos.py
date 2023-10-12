import os
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import pickle
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def get_pos_dict(idd=None,
                 cl_file=os.path.join("..", "data", "pretraining_data", "merge_cl.pt"),
                 yield_file=os.path.join("..", "data", "pretraining_data", "merge_yield.pt"),
                 pos_file=os.path.join("..", "data", "pretraining_data", "merge_pos_dict.pt")
                 ):
    with open(cl_file, "rb") as f:
        cl_data = pickle.load(f)
    with open(yield_file, "rb") as f:
        yield_data = pickle.load(f)
    smiles_set = set()
    for (l, c, r) in cl_data:
        rea = l.split(".") + c.split(".") + r.split(".")
        for s in rea:
            smiles_set.add(s)
    for (l, c, r, y) in yield_data:
        rea = l.split(".") + c.split(".") + r.split(".")
        for s in rea:
            smiles_set.add(s)
    print("Total number of molecules: {}".format(len(smiles_set)))
    # SMILES -> pos
    start_t = time.time()
    n_fail = 0
    smiles_pos_dict = {}
    smiles_list = list(smiles_set)
    # already_smiles_set = set(smiles_pos_dict.keys())
    for i, s in enumerate(smiles_list[idd * 5000:(idd + 1) * 5000]):
        if (i + 1) % 1000 == 0:
            print("{}, time: {}".format(i + 1, time.time() - start_t))
        # if s in already_smiles_set:
        #     continue
        # else:
        try:
            m = Chem.MolFromSmiles(s)
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m)
            m = Chem.RemoveHs(m)
            # Already applied function "trans". Avoid further confusion.
            # new_s = Chem.MolToSmiles(m)
            c = m.GetConformers()[0]
            p = c.GetPositions()
            smiles_pos_dict[s] = p
            # already_smiles_set.add(new_s)
        except:
            n_fail += 1

    with open(pos_file[:-3] + "_{}.pt".format(idd), "wb") as f:
        pickle.dump(smiles_pos_dict, f)
    print("Position num for molecules: {}\n"
          "Fail num: {}".format(len(smiles_pos_dict), n_fail))


def merge_pos_dict(idd=291, pos_file=os.path.join("..", "data", "pretraining_data", "merge_pos_dict.pt")):
    dict_list = []
    for i in range(idd):
        with open(pos_file[:-3] + "_{}.pt".format(i), "rb") as f:
            dict_list.append(pickle.load(f))
    smiles_pos_dict = {k: v for one in dict_list for k, v in one.items()}
    with open(pos_file, "wb") as f:
        pickle.dump(smiles_pos_dict, f)
    print("Merge num: {}".format(len(smiles_pos_dict)))
    print("Check num: {}".format(sum([len(one) for one in dict_list])))


def merge_pos_dict_remaining(idd=270, pos_file=os.path.join("..", "data", "pretraining_data", "merge_pos_dict_r.pt")):
    dict_list = []
    for i in range(idd):
        if i == 246 or i == 265:
            continue
        with open(pos_file[:-3] + "_{}.pt".format(i), "rb") as f:
            dict_list.append(pickle.load(f))
    smiles_pos_dict = {k: v for one in dict_list for k, v in one.items()}
    with open(pos_file, "wb") as f:
        pickle.dump(smiles_pos_dict, f)
    print("Merge num: {}".format(len(smiles_pos_dict)))
    print("Check num: {}".format(sum([len(one) for one in dict_list])))


def get_remaining_pos(idd=None,
                      pos_file=os.path.join("..", "data", "pretraining_data", "merge_pos_dict_r.pt")
                      ):
    with open(os.path.join("..", "data", "pretraining_data", "remaining_smiles.pt"), "rb") as f:
        remaining_smiles = pickle.load(f)
    print("Smiles num: {}".format(len(remaining_smiles)))

    # SMILES -> pos
    start_t = time.time()
    n_fail = 0
    smiles_pos_dict = {}
    smiles_list = remaining_smiles[idd * 2000:(idd + 1) * 2000]
    for i, s in enumerate(smiles_list):
        if (i + 1) % 500 == 0:
            print("{}, time: {}".format(i + 1, time.time() - start_t))
        try:
            m = Chem.MolFromSmiles(s)
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m)
            m = Chem.RemoveHs(m)
            # Already applied function "trans". Avoid further confusion.
            # new_s = Chem.MolToSmiles(m)
            c = m.GetConformers()[0]
            p = c.GetPositions()
            smiles_pos_dict[s] = p
        except:
            n_fail += 1

    with open(pos_file[:-3] + "_{}.pt".format(idd), "wb") as f:
        pickle.dump(smiles_pos_dict, f)
    print("Position num for molecules: {}\n"
          "Fail num: {}".format(len(smiles_pos_dict), n_fail))


def re_get_pos_old(pos_buffer=os.path.join("..", "data", "data_for_yield", "merge_filtered_pos.pt"),
                   pos_new=os.path.join("..", "data", "pretraining_data", "old_pos_dict.pt")):
    with open(pos_buffer, "rb") as f:
        reactions_pos = pickle.load(f)
    smiles_pos_dict = {}
    for one in reactions_pos:
        for s, p in one[:-1]:
            smiles_pos_dict[s] = p
    print("Buffer num: {}".format(len(smiles_pos_dict)))
    n_fail = 0
    start_t = time.time()
    smiles_pos_dict_new = {}
    for i, (s, p) in enumerate(smiles_pos_dict.items()):
        if (i + 1) % 10000 == 0:
            print("{}, time: {}".format(i + 1, time.time() - start_t))
        try:
            m = Chem.MolFromSmiles(s)
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m)
            m = Chem.RemoveHs(m)
            # s_new = Chem.MolToSmiles(m)
            c = m.GetConformers()[0]
            p_new = c.GetPositions()
            smiles_pos_dict_new[s] = p_new
        except:
            smiles_pos_dict_new[s] = p
            n_fail += 1

    with open(pos_new, "wb") as f:
        pickle.dump(smiles_pos_dict_new, f)
    print("Fail num: {}".format(n_fail))


# too slow for new molecules, checked
def re_get_pos_new(cl_file=os.path.join("..", "data", "pretraining_data", "merge_cl.pt"),
                   yield_file=os.path.join("..", "data", "pretraining_data", "merge_yield.pt"),
                   pos_buffer=os.path.join("..", "data", "data_for_yield", "merge_filtered_pos.pt"),
                   pos_file=os.path.join("..", "data", "pretraining_data", "new_pos_dict.pt")):
    with open(pos_buffer, "rb") as f:
        reactions_pos = pickle.load(f)
    smiles_set_old = set()
    for one in reactions_pos:
        for s, p in one[:-1]:
            smiles_set_old.add(s)
    print("Buffer num: {}".format(len(smiles_set_old)))

    with open(cl_file, "rb") as f:
        cl_data = pickle.load(f)
    with open(yield_file, "rb") as f:
        yield_data = pickle.load(f)
    smiles_set = set()
    for (l, c, r) in cl_data:
        rea = l.split(".") + c.split(".") + r.split(".")
        for s in rea:
            smiles_set.add(s)
    for (l, c, r, y) in yield_data:
        rea = l.split(".") + c.split(".") + r.split(".")
        for s in rea:
            smiles_set.add(s)
    print("Total number of molecules: {}".format(len(smiles_set)))
    new_smiles_set = smiles_set.difference(smiles_set_old)
    print("Check num: {}".format(len(new_smiles_set)))

    n_fail = 0
    start_t = time.time()
    smiles_pos_dict_new = {}
    for i, s in enumerate(new_smiles_set):
        if (i + 1) % 10000 == 0:
            print("{}, time: {}".format(i + 1, time.time() - start_t))
        try:
            m = Chem.MolFromSmiles(s)
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m)
            m = Chem.RemoveHs(m)
            # s_new = Chem.MolToSmiles(m)
            c = m.GetConformers()[0]
            p_new = c.GetPositions()
            smiles_pos_dict_new[s] = p_new
        except:
            n_fail += 1

    with open(pos_file, "wb") as f:
        pickle.dump(smiles_pos_dict_new, f)
    print("Fail num: {}".format(n_fail))


if __name__ == '__main__':
    # import sys
    # iddd = int(sys.argv[1])
    # get_pos_dict(idd=iddd)

    # iddd=2
    # pos_file = os.path.join("..", "data", "pretraining_data", "merge_pos_dict.pt")
    # dict_list = []
    # for i in range(iddd):
    #     with open(pos_file[:-3] + "_{}.pt".format(i), "rb") as f:
    #         dict_list.append(pickle.load(f))
    # aaa = dict(**dict_list[0], **dict_list[1])
    # smiles_pos_dict = {k: v for one in dict_list for k, v in one.items()}

    # merge_pos_dict()

    # cl_file = os.path.join("..", "data", "pretraining_data", "merge_cl.pt")
    # yield_file = os.path.join("..", "data", "pretraining_data", "merge_yield.pt")
    # pos_buffer = os.path.join("..", "data", "pretraining_data", "merge_pos_dict.pt")
    # with open(pos_buffer, "rb") as f:
    #     buffer_pos = pickle.load(f)
    # buffer_smiles = set(buffer_pos.keys())
    # print("Buffer num: {}".format(len(buffer_smiles)))
    #
    # with open(cl_file, "rb") as f:
    #     cl_data = pickle.load(f)
    # with open(yield_file, "rb") as f:
    #     yield_data = pickle.load(f)
    # smiles_set = set()
    # for (l, c, r) in cl_data:
    #     rea = l.split(".") + c.split(".") + r.split(".")
    #     for s in rea:
    #         smiles_set.add(s)
    # for (l, c, r, y) in yield_data:
    #     rea = l.split(".") + c.split(".") + r.split(".")
    #     for s in rea:
    #         smiles_set.add(s)
    # print("Total number of molecules: {}".format(len(smiles_set)))
    # remaining_smiles = smiles_set.difference(buffer_smiles)
    # remaining_smiles = list(remaining_smiles)
    # print("Remaining num {}".format(len(remaining_smiles)))
    #
    # with open(os.path.join("..", "data", "pretraining_data", "remaining_smiles.pt"), "wb") as f:
    #     pickle.dump(remaining_smiles, f)

    # import sys
    # iddd = int(sys.argv[1])
    # get_remaining_pos(idd=iddd)

    # merge_pos_dict_remaining()

    # idd = [246, 265]
    # pos_file = os.path.join("..", "data", "pretraining_data", "temp_dict_r.pt")
    # with open(os.path.join("..", "data", "pretraining_data", "remaining_smiles.pt"), "rb") as f:
    #     remaining_smiles = pickle.load(f)
    # print("Smiles num: {}".format(len(remaining_smiles)))
    # n_fail = 0
    # start_t = time.time()
    # smiles_pos_dict, smiles_list = {}, []
    # for i in idd:
    #     smiles_list += remaining_smiles[i * 2000:(i + 1) * 2000]
    # smiles_list = sorted(smiles_list, key=lambda x: len(x))[:3961]
    # for i, s in enumerate(smiles_list):
    #     if (i + 1) % 500 == 0:
    #         print("{}, time: {}".format(i + 1, time.time() - start_t))
    #     try:
    #         m = Chem.MolFromSmiles(s)
    #         m = Chem.AddHs(m)
    #         AllChem.EmbedMolecule(m)
    #         m = Chem.RemoveHs(m)
    #         # Already applied function "trans". Avoid further confusion.
    #         # new_s = Chem.MolToSmiles(m)
    #         c = m.GetConformers()[0]
    #         p = c.GetPositions()
    #         smiles_pos_dict[s] = p
    #     except:
    #         n_fail += 1
    # with open(pos_file, "wb") as f:
    #     pickle.dump(smiles_pos_dict, f)
    # print("Position num for molecules: {}\n"
    #       "Fail num: {}".format(len(smiles_pos_dict), n_fail))

    # file0 = os.path.join("..", "data", "pretraining_data", "temp_dict_r.pt")
    # file1 = os.path.join("..", "data", "pretraining_data", "merge_pos_dict.pt")
    # file2 = os.path.join("..", "data", "pretraining_data", "merge_pos_dict_r.pt")
    # output_file = os.path.join("..", "data", "pretraining_data", "smiles_pos_dict.pt")
    # pos_dict = {}
    # for one in [file0, file1, file2]:
    #     with open(one, "rb") as f:
    #         pos_dict.update(pickle.load(f))
    # with open(output_file, "wb") as f:
    #     pickle.dump(pos_dict, f)
    print()
