"""
reproduce RF, SVM from https://github.com/nsf-c-cas/yield-rxn/blob/master/scripts/ML_models_testing.py
"""
import argparse
from utils_ds import *
import pickle
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as s_rmse
from sklearn.metrics import mean_absolute_error as s_mae
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def generate_s_m_rxns_index(df, mul=0.01):
    rxns = []
    idx_list = []
    for i, row in df.iterrows():
        rxns.append((row['rxn'],) + (row['y'] * 100 * mul,))  # .replace(">>", ".").split(".")
        idx_list.append(row['index'])
    return rxns, idx_list


parser = argparse.ArgumentParser()
parser.add_argument("-dn", "--dataset_name", type=str, default="su", choices=["dy", "az", "su", "dy_reactant"],
                    help="dataset name. Options: az (AstraZeneca),dy (Doyle),su (Suzuki)")
# parser.add_argument("-dp", "--dataset_path", type=str, default='./data/', help="dataset name")
parser.add_argument("-s", "--split", type=str, default="-1")  # default="halide_Br"
parser.add_argument("-rdkit", "--use_rdkit_feats", default='no_rdkit', type=str, help="Use rdkit discriptors or not")
parser.add_argument("-od", "--output_dir", default='results', type=str,
                    help="Output dir for writing features and RF scores")
parser.add_argument("-ne", "--n_estimators", type=float, default=1000, help="Number of trees in RF model")
parser.add_argument("-md", "--max_depth", type=float, default=10, help="Max depth in RF trees")
parser.add_argument("-rs", "--random_state", type=int, default=1, help="Random state for RF model")
# parser.add_argument("-plt", "--plot_yield_dist", type=bool, default=False, help="Plot the yield distribution")
parser.add_argument("-cv", "--cv", type=int, default=5, help="Folds for cross validation")
parser.add_argument("--fine_tuning", action='store_true', help="Fine_tune the superparameters")
parser.add_argument("--no-fine_tuning", action='store_false')
parser.add_argument("-sf", "--Shuffle", default=False, action='store_true', help="Shuffle the reaction and yield")
args = parser.parse_args()
Shuffle = args.Shuffle

data_type = args.dataset_name
split = args.split
use_rdkit_features = args.use_rdkit_feats
ext = '_' + use_rdkit_features
processed = 'processed-0'  # +str(args.random_state)
# inputs
processed_path = os.path.join("/amax/data/shirunhan/reaction/data", data_type if data_type != "dy_reactant" else "dy",
                              processed)

input_data_file = os.path.join(processed_path,
                               ''.join([data_type if data_type != "dy_reactant" else "dy", ext, '.csv']))
# input_split_idx_file = os.path.join(processed_path, 'train_test_idxs.pickle')

# outputs
output_path = os.path.join(data_type, split)
if not os.path.exists(output_path):
    os.makedirs(output_path)

r2_fn = os.path.join(output_path, 'results_r2' + ext + '_1000.csv')
mae_fn = os.path.join(output_path, 'results_mae' + ext + '_1000.csv')
rmse_fn = os.path.join(output_path, 'results_rmse' + ext + '_1000.csv')

print("\n\nReading data from: ", input_data_file)
print("Using rdkit features!") if use_rdkit_features == 'rdkit' else print("Not using rdkit features!")

df = pd.read_csv(input_data_file, index_col=0)

smiles_features = ["reactant_smiles", "solvent_smiles", "base_smiles", "product_smiles"]
"""check the split, right or not"""
from rdkit import Chem

reactants_smiles = []
for one in df["reactant_smiles"]:
    reactants_smiles.append([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in one.split(".")])
df.drop(smiles_features, axis=1, inplace=True)

print(f"Raw data frame shape: {df.shape}")


# if args.plot_yield_dist:
#     print(f"Plotting yield distibution:")
#     df['yield'].plot(kind='hist', bins=12)


def split_scale_data(df, training_idx, test_idx, label_name, seed=1):
    """
    split the raw data into train and test using
    pre-writtten indexes. Then standardize the train
    and test set.
    """

    train_set = df.iloc[training_idx]
    test_set = df.iloc[test_idx]

    train_set.pop('id'), test_set.pop('id')
    y_train, y_test = train_set.pop(label_name), test_set.pop(label_name)
    x_train, x_test = train_set, test_set

    if args.Shuffle:
        y_train = shuffle(y_train, random_state=seed)
        y_test = shuffle(y_test, random_state=seed)
        print('shuffle finished')

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    return x_train_scaled, x_test_scaled, y_train, y_test


def get_sorted_feat_importances(feat_names, feat_importances):
    """
    sort the feature names based on RF feature importances
    and return the sorted feat names as well as pair:
    (feat_name, score)
    """
    sorted_idx = (feat_importances).argsort()  # [:n]

    sorted_feat_names = [feat_names[i] for i in sorted_idx]
    sorted_feat_importances = feat_importances[sorted_idx]
    final_feat_importances = list(zip(sorted_feat_names, sorted_feat_importances))

    return sorted_feat_names, final_feat_importances


selected_features = set()
result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# with open(input_split_idx_file, 'rb') as handle:
#     idx_dict = pickle.load(handle)

data_prefix = ""
if data_type == "dy_reactant":
    """these idxs are generated by ourselves using FullCV_01, 
    but the data with feature is provided by Plate1-3 indexes"""
    with open(os.path.join(data_prefix, "data", "BH/reactant_split_idxs.pickle"), "rb") as f:
        train_test_idxs = pickle.load(f)
    with open(os.path.join(data_prefix, "data", "BH/fullcv01_to_plate1-3.pkl"), "rb") as f:
        s2t = pickle.load(f)
    training_data_idx = np.array([s2t[one] for one in train_test_idxs[split]["train_idx"]])
    test_data_idx = np.array([s2t[one] for one in train_test_idxs[split]["test_idx"]])

    """test Br, right or not"""
    Br = ['COc1ccc(Br)cc1', 'CCc1ccc(Br)cc1', 'Brc1cccnc1', 'Brc1ccccn1', 'FC(F)(F)c1ccc(Br)cc1']
    test_right = []
    for id_, one in enumerate(reactants_smiles):
        for s in Br:
            if s in one:
                test_right.append(id_)
                break
    print(set(list(test_data_idx)) == set(test_right))
    print()
elif data_type == "dy":
    """BH"""
    # test range
    name_split_dict = {"BH_test1": ('Test1', 3058 - 1, 3955), "BH_test2": ('Test2', 3056 - 1, 3955),
                       "BH_test3": ('Test3', 3059 - 1, 3955), "BH_test4": ('Test4', 3056 - 1, 3955),
                       "BH_plate1": ('Plates1-3', 1 - 1, 1075), "BH_plate2": ('Plates1-3', 1076 - 1, 2515),
                       "BH_plate3": ('Plates1-3', 2516 - 1, 3955), "BH_plate2_new": ('Plate2_new', 1076 - 1, 2515)}

    (name, start, end) = name_split_dict[split]
    df_doyle_base = pd.read_excel(os.path.join(data_prefix, 'data', 'BH/Dreher_and_Doyle_input_data.xlsx'),
                                  sheet_name="Plates1-3", engine='openpyxl')
    df_doyle_base['rxn'] = generate_buchwald_hartwig_rxns(df_doyle_base, 0.01)
    data2idx = {}
    for idx, one in enumerate(df_doyle_base['rxn']):
        data2idx[one] = idx

    # # FullCV_01
    # name_split = [('FullCV_{:02d}'.format(i), int(3955*0.7)) for i in range(1, 11)]
    # name, start = name_split[0]
    # end = 3955

    df_doyle = pd.read_excel(os.path.join(data_prefix, 'data', 'BH/Dreher_and_Doyle_input_data.xlsx'),
                             sheet_name=name, engine='openpyxl')
    df_doyle['rxn'] = generate_buchwald_hartwig_rxns(df_doyle, 0.01)
    dataset_idx = []
    for one in df_doyle['rxn']:
        dataset_idx.append(data2idx[one])
    training_data_idx = dataset_idx[:start] + dataset_idx[end:]
    test_data_idx = dataset_idx[start:end]
elif data_type == "su":
    """SM, 4543??"""
    # test range
    name_split_dict = {"SM_test1": ('1', 4320, 5760), "SM_test2": ('2', 4320, 5760),
                       "SM_test3": ('3', 4320, 5760), "SM_test4": ('4', 4320, 5760)}

    (name, start, end) = name_split_dict[split]
    df_su_base = pd.read_csv(os.path.join(data_prefix, 'data', 'SM/random_split_0_custom.tsv'), sep='\t')
    SM_dataset, idx_list = generate_s_m_rxns_index(df_su_base, 0.01)
    data2idx = {}
    for idx, one in enumerate(SM_dataset):
        data2idx[one] = idx_list[idx]

    df_su = pd.read_csv(os.path.join(data_prefix, 'data', 'SM/SM_Test_{}.tsv'.format(name)), sep='\t')
    raw_dataset = generate_s_m_rxns(df_su, 0.01)
    dataset_idx = []
    for one in raw_dataset:
        dataset_idx.append(data2idx[one])
    training_data_idx = dataset_idx[:start]
    test_data_idx = dataset_idx[start:]
elif data_type == "az":
    # RandomForestRegressor(max_depth=20, min_samples_leaf=2, n_estimators=500,
    #                   random_state=1)
    with open(os.path.join(data_prefix, "data", "az", "processed-0", "train_test_idxs.pickle"), "rb") as f:
        train_test_idxs = pickle.load(f)
    print()
else:
    training_data_idx = []
    test_data_idx = []
    print(f"Wrong [{data_type}]")
    exit()

if args.fine_tuning:
    print('start fine-tuning')

    """a little 'strange', the same split is OK, what about different splits?"""
    if data_type == "az":
        training_data_idx_, test_data_idx_ = shuffle(train_test_idxs["train_idx"][1], random_state=1), shuffle(
            train_test_idxs["test_idx"][1], random_state=1)
    else:
        training_data_idx_, test_data_idx_ = shuffle(training_data_idx, random_state=1), shuffle(test_data_idx,
                                                                                                 random_state=1)
    x_train_scaled, x_test_scaled, y_train, y_test = split_scale_data(df, training_data_idx_, test_data_idx_, 'yield')

    rf = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state, max_depth=args.max_depth)
    hyperpara = [{'n_estimators': [100, 200, 500], 'max_depth': [10, 15, 20],
                  "min_samples_leaf": [2, 4, 6], "min_samples_split": [1, 2, 4]}]
    rf = GridSearchCV(rf, hyperpara, cv=args.cv)
    # cv svr here, too?
    svr = svm.SVR(kernel="rbf")
    param_dict = [{"kernel": ["rbf"],
                   "gamma": [2 * 10 ** i for i in range(-5, 1)],
                   "C": [10 ** i for i in range(-2, 3)]}]
    svr = GridSearchCV(svr, param_dict, cv=args.cv)

    y_test, y_train = y_test / 100, y_train / 100

    sh1 = rf.fit(x_train_scaled, y_train)
    print(sh1.best_estimator_)
    sh2 = svr.fit(x_train_scaled, y_train)
    print(sh2.best_estimator_)
    print('parameter fine-tuning done')

y_results = {"train": [], "test": []}
for split_set_num in range(1, 5 + 1 if data_type != "az" else 10 + 1):
    seed_torch(split_set_num)
    result_dict['r2'][split_set_num]['model_num'] = split_set_num
    result_dict['mae'][split_set_num]['model_num'] = split_set_num
    result_dict['rmse'][split_set_num]['model_num'] = split_set_num

    if data_type == "az":
        training_data_idx_, test_data_idx_ = shuffle(train_test_idxs["train_idx"][split_set_num],
                                                     random_state=split_set_num), shuffle(
            train_test_idxs["test_idx"][split_set_num], random_state=split_set_num)
    else:
        training_data_idx_, test_data_idx_ = shuffle(training_data_idx, random_state=split_set_num), shuffle(
            test_data_idx, random_state=split_set_num)
    x_train_scaled, x_test_scaled, y_train, y_test = split_scale_data(df, training_data_idx_, test_data_idx_, 'yield')
    y_test, y_train = y_test / 100, y_train / 100

    rf = sh1.best_estimator_ if args.fine_tuning else RandomForestRegressor(n_estimators=args.n_estimators,
                                                                            max_depth=args.max_depth)
    # svm uses cv 5 by default? why "do not" use CV to select hyper-parameters?
    SVM = sh2.best_estimator_ if args.fine_tuning else svm.SVR()
    seed_torch(split_set_num)

    rf.fit(x_train_scaled, y_train)
    SVM.fit(x_train_scaled, y_train)

    rf_y_pred_train = rf.predict(x_train_scaled)
    rf_y_pred_test = rf.predict(x_test_scaled)

    SVM_y_pred_train = SVM.predict(x_train_scaled)
    SVM_y_pred_test = SVM.predict(x_test_scaled)

    train_r2 = round(r2_score(y_train, rf_y_pred_train), 4)
    test_r2 = round(r2_score(y_test, rf_y_pred_test), 4)

    result_dict['r2_rf'][split_set_num]['train'] = round(r2_score(y_train, rf_y_pred_train), 4)
    result_dict['r2_rf'][split_set_num]['test'] = round(r2_score(y_test, rf_y_pred_test), 4)

    result_dict['r2_svm'][split_set_num]['train'] = round(r2_score(y_train, SVM_y_pred_train), 4)
    result_dict['r2_svm'][split_set_num]['test'] = round(r2_score(y_test, SVM_y_pred_test), 4)

    result_dict['mae_rf'][split_set_num]['train'] = round(s_mae(y_train, rf_y_pred_train), 4)
    result_dict['mae_rf'][split_set_num]['test'] = round(s_mae(y_test, rf_y_pred_test), 4)

    result_dict['mae_svm'][split_set_num]['train'] = round(s_mae(y_train, SVM_y_pred_train), 4)
    result_dict['mae_svm'][split_set_num]['test'] = round(s_mae(y_test, SVM_y_pred_test), 4)

    result_dict['rmse_rf'][split_set_num]['train'] = round(s_rmse(y_train, rf_y_pred_train, squared=False), 4)
    result_dict['rmse_rf'][split_set_num]['test'] = round(s_rmse(y_test, rf_y_pred_test, squared=False), 4)

    result_dict['rmse_svm'][split_set_num]['train'] = round(s_rmse(y_train, SVM_y_pred_train, squared=False), 4)
    result_dict['rmse_svm'][split_set_num]['test'] = round(s_rmse(y_test, SVM_y_pred_test, squared=False), 4)

    print(f'\nModel number: {split_set_num}')
    print(f'Mean Absolute Train R2: ', train_r2)
    print(f'Mean Absolute Test R2: ', test_r2)
    y_results["train"].append((y_train, rf_y_pred_train))
    y_results["test"].append((y_test, rf_y_pred_test))
with open(os.path.join(output_path, "ys.pkl"), "wb") as f:
    pickle.dump((sh1.best_estimator_ if args.fine_tuning else RandomForestRegressor(n_estimators=args.n_estimators,
                                                                                    max_depth=args.max_depth),
                 y_results), f)

dp_r2_rf = pd.DataFrame.from_dict(result_dict['r2_rf'], orient='index',
                                  columns=['model_num', 'train', 'test'])
summary = dp_r2_rf.describe().loc[['mean', 'std']]
dp_r2_rf = pd.concat([dp_r2_rf, summary], axis=0)
dp_r2_svm = pd.DataFrame.from_dict(result_dict['r2_svm'], orient='index',
                                   columns=['model_num', 'train', 'test'])
summary = dp_r2_svm.describe().loc[['mean', 'std']]
dp_r2_svm = pd.concat([dp_r2_svm, summary], axis=0)
dp_r2 = pd.concat([dp_r2_rf, dp_r2_svm], axis=1, join='inner')

dp_mae_rf = pd.DataFrame.from_dict(result_dict['mae_rf'], orient='index',
                                   columns=['model_num', 'train', 'test'])
summary = dp_mae_rf.describe().loc[['mean', 'std']]
dp_mae_rf = pd.concat([dp_mae_rf, summary], axis=0)
dp_mae_svm = pd.DataFrame.from_dict(result_dict['mae_svm'], orient='index',
                                    columns=['model_num', 'train', 'test'])
summary = dp_mae_svm.describe().loc[['mean', 'std']]
dp_mae_svm = pd.concat([dp_mae_svm, summary], axis=0)
dp_mae = pd.concat([dp_mae_rf, dp_mae_svm], axis=1, join='inner')

dp_rmse_rf = pd.DataFrame.from_dict(result_dict['rmse_rf'], orient='index',
                                    columns=['model_num', 'train', 'test'])
summary = dp_rmse_rf.describe().loc[['mean', 'std']]
dp_rmse_rf = pd.concat([dp_rmse_rf, summary], axis=0)
dp_rmse_svm = pd.DataFrame.from_dict(result_dict['rmse_svm'], orient='index',
                                     columns=['model_num', 'train', 'test'])
summary = dp_rmse_svm.describe().loc[['mean', 'std']]
dp_rmse_svm = pd.concat([dp_rmse_svm, summary], axis=0)
dp_rmse = pd.concat([dp_rmse_rf, dp_rmse_svm], axis=1, join='inner')

dp_r2.to_csv(r2_fn)
dp_mae.to_csv(mae_fn)
dp_rmse.to_csv(rmse_fn)
