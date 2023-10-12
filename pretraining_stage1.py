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
assert args.supervised == 0
assert args.data_type == "rnn_geo"
model_save_dir = os.path.join("pretraining", args.data_type)
for sub_name in ["log", "run", "model"]:
    sub_dir = os.path.join(model_save_dir, sub_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
args.log_path = "stage1_{}_{}_{}_{}_{}".format(args.batch_size, args.lr, args.lr_type, args.cl_weight, args.kl_weight)
logger = create_file_logger(os.path.join(model_save_dir, "log", "{}.txt".format(args.log_path)))
writer = SummaryWriter(os.path.join(model_save_dir, "run", "{}".format(args.log_path)))


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
    datasets = [MyDataset(datasets[i]) for i in range(len(datasets))]

    def collate_fn_3d(data_, use_3d=True, radius=10.0):
        dgl_pool, dgl_len_pool = [], []
        smiles_index_pool, smiles_index_len_pool = [], []
        for idd, (l, c, r, reaction, one_index_list) in enumerate(data_):
            smiles_index_pool.append(torch.as_tensor(one_index_list, dtype=torch.long))
            smiles_index_len_pool.append(len(one_index_list))
            for s in reaction:
                dgl_pool.append(mol_to_dgl_graph(Chem.MolFromSmiles(s), use_3d, smiles_pos_dict[s], radius))
            dgl_len_pool.append(len(reaction))
        return rnn_utils.pad_sequence(smiles_index_pool, batch_first=True, padding_value=0), \
               torch.as_tensor(smiles_index_len_pool), \
               dgl.batch(dgl_pool), torch.as_tensor(dgl_len_pool)

    training_dataloader = DataLoader(datasets[0], batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                     collate_fn=collate_fn_3d, num_workers=16)
    validation_dataloader = DataLoader(datasets[1], batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                       collate_fn=collate_fn_3d, num_workers=16)
    test_dataloader = DataLoader(datasets[2], batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                 collate_fn=collate_fn_3d, num_workers=16)

    pre_model = ReactionCLModel(args)

    cl_loss_fc = dual_CL

    best_p = [1e3 for _ in range(3)]
    training_step, validation_step, test_step = 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        # training_dataloader = get_data_loader_cl(datasets[0], smiles_pos_dict,
        #                                          use_3d=True, radius=10.0, batch_size=args.batch_size)
        # validation_dataloader = get_data_loader_cl(datasets[1], smiles_pos_dict,
        #                                            use_3d=True, radius=10.0, batch_size=args.batch_size)
        # test_dataloader = get_data_loader_cl(datasets[2], smiles_pos_dict,
        #                                      use_3d=True, radius=10.0, batch_size=args.batch_size)
        training_loss, training_t, training_step = \
            pre_model.training(training_dataloader, epoch,
                               training_step, "training", logger, writer, cl_loss_fc, args.metric)
        validation_loss, validation_t, validation_step = \
            pre_model.training(validation_dataloader, epoch,
                               validation_step, "validation", logger, writer, cl_loss_fc, args.metric)
        test_loss, test_t, test_step = \
            pre_model.training(test_dataloader, epoch,
                               test_step, "test", logger, writer, cl_loss_fc, args.metric)
        pre_model.scheduler.step()
        logger.info("epoch {}, loss training: {:.6f}, loss validation: {:.6f}, loss test: {:.6f}, time: {:.2f}".
                    format(epoch, training_loss, validation_loss, test_loss, training_t + validation_t + test_t))
        is_best = validation_loss < best_p[1]
        if validation_loss < best_p[1]:
            best_p = [training_loss, validation_loss, test_loss]
        pre_model.save(epoch, best_p, is_best, model_save_dir)
    logger.info("Training: {}".format(best_p[0]))
    logger.info("Validation: {}".format(best_p[1]))
    logger.info("Test: {}".format(best_p[2]))
    writer.close()


if __name__ == '__main__':
    main()
