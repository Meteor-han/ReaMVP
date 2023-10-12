import pickle

from torch.utils.tensorboard import SummaryWriter
from models.reaction import *
import time
import matplotlib as mpl
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')
mpl.use('Agg')

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

args = get_args()
assert args.supervised in [1, 2]
model_save_dir = os.path.join("pretraining", args.data_type)
for sub_name in ["log", "run", "model"]:
    sub_dir = os.path.join(model_save_dir, sub_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
args.log_path = "stage2_{}_{}_{}_{}_{}_{}" \
    .format(args.batch_size, args.lr, args.lr_type, args.loss_type, args.cl_weight, args.kl_weight)
logger = create_file_logger(os.path.join(model_save_dir, "log", "{}_{}.txt".format(args.log_path, args.supervised)))
if args.save == 1:
    writer = SummaryWriter(os.path.join(model_save_dir, "run", "{}_{}".format(args.log_path, args.supervised)))
else:
    writer = None


def main():
    logger.info(f"======={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")
    logger.info("=======Setting=======")
    for k in args.__dict__:
        v = args.__dict__[k]
        logger.info("{}: {}".format(k, v))
    logger.info("=======Pretraining=======")
    seed_torch(args.seed)

    args.vocab_size = 142
    args.device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"

    with open(args.pos_dict_path, "rb") as f:
        smiles_pos_dict = pickle.load(f)
    # shuffled input
    datasets = []
    for name in ["training", "validation", "test"]:
        with open(args.data_path + "_{}.pt".format(name), "rb") as f:
            datasets.append(pickle.load(f))
    # normalize y
    ys = np.zeros(len(datasets[0]))
    for i, (l, c, r, reaction, one_index_list, y) in enumerate(datasets[0]):
        ys[i] = y / 100
    # mean-std, min-max
    if args.normalize == 1:
        args.training_mean = np.mean(ys)
        args.training_std = np.std(ys)
    elif args.normalize == 2:
        args.training_mean = np.min(ys)
        args.training_std = np.max(ys) - np.min(ys)
    else:
        args.training_mean = 0.
        args.training_std = 1.
    logger.info("Mean: {:.4f}, Std: {:.4f}".format(args.training_mean, args.training_std))

    geo_tag = False if args.data_type == "rnn" else True

    def collate_fn_3d_y(data_, use_3d=True, radius=10.0, geo=geo_tag):
        dgl_pool, dgl_len_pool, target_pool = [], [], []
        smiles_index_pool, smiles_index_len_pool = [], []
        for idd, (l, c, r, reaction, one_index_list, y) in enumerate(data_):
            smiles_index_pool.append(torch.tensor(one_index_list, dtype=torch.long))
            smiles_index_len_pool.append(len(one_index_list))
            if geo:
                # still contains ""?
                temp_n = len(reaction)
                for s in reaction:
                    if not s:
                        temp_n -= 1
                        continue
                    dgl_pool.append(mol_to_dgl_graph(Chem.MolFromSmiles(s), use_3d, smiles_pos_dict[s], radius))
                dgl_len_pool.append(temp_n)
            # not do forget... 100
            target_pool.append(y / 100)
            if not geo:
                dgl_pool.append(dgl.graph(([0], [1])))
                dgl_len_pool.append(1)
        return rnn_utils.pad_sequence(smiles_index_pool, batch_first=True, padding_value=0), \
               torch.tensor(smiles_index_len_pool), \
               dgl.batch(dgl_pool), torch.tensor(dgl_len_pool), \
               torch.tensor(target_pool, dtype=torch.float).unsqueeze(-1)

    datasets = [MyDataset(datasets[i]) for i in range(len(datasets))]
    training_dataloader = DataLoader(datasets[0], batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                     collate_fn=collate_fn_3d_y, num_workers=16)
    validation_dataloader = DataLoader(datasets[1], batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                       collate_fn=collate_fn_3d_y, num_workers=16)
    test_dataloader = DataLoader(datasets[2], batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                 collate_fn=collate_fn_3d_y, num_workers=16)

    pre_model = ReactionSLModel(args=args)
    if args.supervised == 1:
        pretraining_state_dict = torch.load(
            os.path.join("pretraining", "rnn_geo", "model", "stage1_256_0.001_cos_{}_{}_best.pt"
                         .format(args.cl_weight, args.kl_weight)), map_location=args.device)
        pre_model.load_state_dict(pretraining_state_dict)

    loss_fc = nn.MSELoss() if args.loss_type == "mse" else nn.L1Loss()
    mse_fc = nn.MSELoss()
    mae_fc = nn.L1Loss()

    best_p = [[1e3, 1e3, 0] for _ in range(3)]
    training_step, validation_step, test_step = 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        # training_dataloader = get_data_loader_y(datasets[0], smiles_pos_dict,
        #                                         use_3d=True, radius=10.0, batch_size=args.batch_size, geo=geo_tag)
        # validation_dataloader = get_data_loader_y(datasets[1], smiles_pos_dict,
        #                                           use_3d=True, radius=10.0, batch_size=args.batch_size, geo=geo_tag)
        # test_dataloader = get_data_loader_y(datasets[2], smiles_pos_dict,
        #                                     use_3d=True, radius=10.0, batch_size=args.batch_size, geo=geo_tag)
        training_loss, training_t, training_step, _, _ = \
            pre_model.training(training_dataloader, epoch,
                               training_step, "training", logger, writer, loss_fc, mse_fc, mae_fc, eval_every=200)
        validation_loss, validation_t, validation_step, _, _ = \
            pre_model.training(validation_dataloader, epoch,
                               validation_step, "validation", logger, writer, loss_fc, mse_fc, mae_fc, eval_every=200)
        test_loss, test_t, test_step, _, _ = \
            pre_model.training(test_dataloader, epoch,
                               test_step, "test", logger, writer, loss_fc, mse_fc, mae_fc, eval_every=200)
        pre_model.scheduler.step()
        logger.info("epoch {}, loss training: {:.6f}, loss validation: {:.6f}, loss test: {:.6f}, time: {:.2f}".
                    format(epoch, training_loss[0], validation_loss[0], test_loss[0],
                           training_t + validation_t + test_t))
        is_best = validation_loss[0] < best_p[1][0]
        if validation_loss[0] < best_p[1][0]:
            best_p = [training_loss, validation_loss, test_loss]
        if args.save == 1:
            pre_model.save(epoch, best_p, is_best, model_save_dir)
    # transformed_p = [np.sqrt(best_p[0][0]) * args.training_std,
    #                  np.sqrt(best_p[1][0]) * args.training_std,
    #                  np.sqrt(best_p[2][0]) * args.training_std,
    #                  best_p[0][1] * args.training_std,
    #                  best_p[1][1] * args.training_std,
    #                  best_p[2][1] * args.training_std]
    # logger.info("Training, Validation, Test:\n{}".format(transformed_p))
    # RMSE, MAE, R2
    yield_ratio = 1.
    transformed_p = [yield_ratio * best_p[0][0], yield_ratio * best_p[0][1], best_p[0][2],
                     yield_ratio * best_p[1][0], yield_ratio * best_p[1][1], best_p[1][2],
                     yield_ratio * best_p[2][0], yield_ratio * best_p[2][1], best_p[2][2]]
    logger.info("RMSE, MAE, R2:\n{}\n{}\n{}".format(transformed_p[:3], transformed_p[3:6], transformed_p[6:]))
    if args.save == 1:
        writer.close()


if __name__ == '__main__':
    main()
