from utils_ds import *
from models.reaction import *
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print("tensorboard, fail")
import pickle
from copy import deepcopy
import matplotlib as mpl
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')
mpl.use('Agg')

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


class Finetuner:
    def __init__(self):
        args = get_args()
        self.model_save_dir = os.path.join("downstream", args.ds, args.data_type)
        for sub_name in ["log", "run", "model"]:
            sub_dir = os.path.join(self.model_save_dir, sub_name)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
        log_tag = "{}".format(args.split[0]) if args.ds in ["BH", "SM"] else "log"
        args.log_path = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}". \
            format(log_tag, args.batch_size, args.lr, args.lr_type, args.epochs,
                   args.weight_decay, args.predictor_dropout, args.normalize, args.predictor_bn,
                   args.predictor_num_layers, args.loss_type, args.mlp_only, args.cl_weight, args.kl_weight)
        self.logger = create_file_logger(
            os.path.join(self.model_save_dir, "log", "{}_{}.txt".format(args.log_path, args.supervised)))
        if args.save == 1:
            self.writer = SummaryWriter(os.path.join(self.model_save_dir, "run",
                                                     "{}_{}".format(args.log_path, args.supervised)))
        else:
            self.writer = None
        self.logger.info(f"======={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")
        self.logger.info("=======Setting=======")
        for k in args.__dict__:
            v = args.__dict__[k]
            self.logger.info("{}: {}".format(k, v))
        self.logger.info("=======Training=======")
        seed_torch(args.seed)

        args.vocab_size = 142
        args.device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"

        self.args = args
        self.loss_fc = nn.MSELoss() if args.loss_type == "mse" else nn.L1Loss()
        self.mse_fc = nn.MSELoss()
        self.mae_fc = nn.L1Loss()

    def print_results(self, p):
        all_best_p = np.concatenate(p, axis=0)
        self.logger.info("All:")
        self.logger.info(all_best_p)
        self.logger.info("Mean and std:")
        self.logger.info(np.mean(all_best_p, axis=0).tolist())
        self.logger.info(np.std(all_best_p, axis=0).tolist())

    def train(self, training_data, test_data, yield_ratio=100.):
        # normalize y; worse?
        ys = np.zeros(len(training_data))
        for i, (_, one_index_list, one_dgl_list, y) in enumerate(training_data):
            ys[i] = y
        # mean-std, min-max
        if self.args.normalize == 1:
            self.args.training_mean = np.mean(ys)
            self.args.training_std = np.std(ys)
        elif self.args.normalize == 2:
            self.args.training_mean = np.min(ys)
            self.args.training_std = np.max(ys) - np.min(ys)
        else:
            self.args.training_mean = 0.
            self.args.training_std = 1.
        self.logger.info("Mean: {:.4f}, Std: {:.4f}".format(self.args.training_mean, self.args.training_std))

        training_data = MyDataset(training_data)
        test_data = MyDataset(test_data)
        training_loader = DataLoader(training_data, batch_size=self.args.batch_size,
                                     shuffle=True, collate_fn=my_collate_fn_3d, drop_last=False, num_workers=self.args.data_workers)
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size,
                                 shuffle=False, collate_fn=my_collate_fn_3d, drop_last=False, num_workers=self.args.data_workers)

        # load model
        ft_model = FinetuneModel(args=self.args)
        if self.args.supervised == 0:
            pretraining_state_dict = torch.load(
                os.path.join("checkpoint", "stage1_256_0.001_cos_{}_{}_best.pt"
                             .format(self.args.cl_weight, self.args.kl_weight)), map_location=self.args.device)
        elif self.args.supervised == 1:
            pretraining_state_dict = torch.load(
                os.path.join("checkpoint", f"{self.args.data_type}", "stage2_256_0.001_cos_mse_{}_{}_1_best.pt"
                             .format(self.args.cl_weight, self.args.kl_weight)), map_location=self.args.device)
            # pretraining_state_dict["predictor_state_dict"] = temp_state_dict["predictor_state_dict"]
        else:
            pretraining_state_dict = None
            print("Not yet")
            exit()
        ft_model.load_state_dict(pretraining_state_dict)

        best_p = [[1e3, 1e3, 1e3, 0] for _ in range(2)]
        training_step, test_step = 0, 0
        for epoch in range(1, self.args.epochs + 1):
            training_loss, training_t, training_step, _, _ = \
                ft_model.training(training_loader, epoch, training_step, "training", self.logger, self.writer,
                                  self.loss_fc, self.mse_fc, self.mae_fc)
            test_loss, test_t, test_step, _, _ = \
                ft_model.training(test_loader, epoch, test_step, "test", self.logger, self.writer,
                                  self.loss_fc, self.mse_fc, self.mae_fc)
            ft_model.scheduler.step()
            self.logger.info("epoch {}, loss training: {:.6f}, loss test: {:.6f}, time: {:.2f}".
                             format(epoch, training_loss[0], test_loss[0], training_t + test_t))
            is_best = test_loss[0] < best_p[1][0]
            if test_loss[0] < best_p[1][0]:
                best_p = [training_loss, test_loss]
            if self.args.save == 1:
                ft_model.save(epoch, best_p, is_best, self.model_save_dir)
        # RMSE, MAE, R2
        transformed_p = [yield_ratio * best_p[0][1], yield_ratio * best_p[0][2], best_p[0][3],
                         yield_ratio * best_p[1][1], yield_ratio * best_p[1][2], best_p[1][3]]
        self.logger.info("RMSE, MAE, R2:\n{}".format(transformed_p))
        if self.args.save == 1:
            self.writer.close()
        return np.array([transformed_p])

    def run_BH_xx(self):
        # test range
        name_split_dict = {"BH_test1": ('Test1', 3058 - 1, 3955), "BH_test2": ('Test2', 3056 - 1, 3955),
                           "BH_test3": ('Test3', 3059 - 1, 3955), "BH_test4": ('Test4', 3056 - 1, 3955),
                           "BH_plate1": ('Plates1-3', 1 - 1, 1075), "BH_plate2": ('Plates1-3', 1076 - 1, 2515),
                           "BH_plate3": ('Plates1-3', 2516 - 1, 3955), "BH_plate2_new": ('Plate2_new', 1076 - 1, 2515)}

        (name, start, end) = name_split_dict[self.args.ds]
        self.args.name = name
        all_best_p = []
        data_path = os.path.join("data", "BH/BH_index_dgl_dict.pt")
        with open(data_path, "rb") as f:
            original_dataset, _, _ = pickle.load(f)
        dataset_dict = {}
        for key, one_index_list, one_dgl_list, y in original_dataset:
            dataset_dict[key] = [key, one_index_list, one_dgl_list, y]
        # 5 random initialization
        for seed in range(self.args.repeat):
            seed_torch(seed)
            df_doyle = pd.read_excel(os.path.join('data', 'BH/Dreher_and_Doyle_input_data.xlsx'),
                                     sheet_name=name, engine='openpyxl')
            df_doyle['rxn'] = generate_buchwald_hartwig_rxns(df_doyle, 0.01)
            dataset = []
            for one in df_doyle['rxn']:
                dataset.append(dataset_dict[one])
            training_data = dataset[:start] + dataset[end:]
            test_data = dataset[start:end]

            p = self.train(training_data, test_data)
            all_best_p.append(p)
        self.print_results(all_best_p)

    def run_SM_xx(self):
        # test range
        name_split_dict = {"SM_test1": ('1', 4320, 5760), "SM_test2": ('2', 4320, 5760),
                           "SM_test3": ('3', 4320, 5760), "SM_test4": ('4', 4320, 5760)}

        (name, start, end) = name_split_dict[self.args.ds]
        self.args.name = name
        all_best_p = []
        data_path = os.path.join("data", "SM/SM_index_dgl_dict.pt")
        with open(data_path, "rb") as f:
            original_dataset, _, _ = pickle.load(f)
        dataset_dict = {}
        for key, one_index_list, one_dgl_list, y in original_dataset:
            dataset_dict[key] = [key, one_index_list, one_dgl_list, y]
        # 5 random initialization
        for seed in range(self.args.repeat):
            seed_torch(seed)
            df = pd.read_csv(os.path.join('data', 'SM/SM_Test_{}.tsv'.format(name)),
                             sep='\t')
            raw_dataset = generate_s_m_rxns(df, 0.01)
            dataset = []
            for one in raw_dataset:
                dataset.append(dataset_dict[one])
            training_data = dataset[:start]
            test_data = dataset[start:]

            p = self.train(training_data, test_data)
            all_best_p.append(p)
        self.print_results(all_best_p)

    def run_BH_or_SM(self):
        """add split ratio"""
        if self.args.ds == "BH":
            num_ = int(3955 * self.args.split[0])
            name_split = [('FullCV_{:02d}'.format(i), num_) for i in range(1, 11)]
            data_path = os.path.join("data", "{}/{}_index_dgl_dict.pt".format(self.args.ds, self.args.ds))
        else:
            num_ = int(5760 * self.args.split[0])
            name_split = [('random_split_{}'.format(i), num_) for i in range(10)]
            data_path = os.path.join("data",
                                     "{}/{}_index_dgl_dict.pt".format(self.args.ds, self.args.ds))
        with open(data_path, "rb") as f:
            original_dataset, _, _ = pickle.load(f)
        dataset_dict = {}
        for key, one_index_list, one_dgl_list, y in original_dataset:
            dataset_dict[key] = [key, one_index_list, one_dgl_list, y]

        all_best_p = []
        # 10 splits
        for i, (name, split) in enumerate(name_split):
            # for parameter selection
            if i >= self.args.repeat:
                break
            if self.args.ds == "BH":
                df_doyle = pd.read_excel(os.path.join('data', 'BH/Dreher_and_Doyle_input_data.xlsx'),
                                         sheet_name=name, engine='openpyxl')
                raw_dataset = generate_buchwald_hartwig_rxns(df_doyle, 0.01)
            else:
                df = pd.read_csv(os.path.join('data', 'SM/{}.tsv'.format(name)), sep='\t')
                raw_dataset = generate_s_m_rxns(df, 0.01)
            dataset = []
            for one in raw_dataset:
                dataset.append(dataset_dict[one])
            training_data = dataset[:split]
            test_data = dataset[split:]

            p = self.train(training_data, test_data)
            all_best_p.append(p)
        self.print_results(all_best_p)


if __name__ == '__main__':
    runer = Finetuner()
    # try:
    if True:
        if runer.args.ds in ["BH", "SM"]:
            runer.run_BH_or_SM()
        elif runer.args.ds in ["SM_test1", "SM_test2", "SM_test3", "SM_test4"]:
            runer.run_SM_xx()
        else:
            runer.run_BH_xx()
    # except:
    #     runer.logger.info("Fail")
