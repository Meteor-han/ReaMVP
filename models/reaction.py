from models.rnn import *
from models.schnet import *
from utils import *
from torch import optim
import time
from sklearn.metrics import r2_score


class ReactionCLModel:
    def __init__(self, args=None):
        self.args = args
        multi = 2 if args.rnn_use_bidirectional else 1
        self.Projector = MLP(input_dim=args.rnn_hidden_dim * multi,
                             hidden_dim=args.rnn_hidden_dim * multi,
                             output_dim=args.rnn_hidden_dim * multi,
                             num_layers=2, dropout=0, bn=False).to(args.device)
        if args.smiles_use_lstm:
            self.SMILESRNNModel = EmbedLSTMSMILES(args.vocab_size, args.smiles_embed_size, args.smiles_hidden_dim,
                                                  args.smiles_use_bidirectional, args.smiles_n_layer,
                                                  args.smiles_dropout, args.smiles_use_lstm).to(args.device)
        else:
            self.SMILESRNNModel = EmbedGRUSMILES(args.vocab_size, args.smiles_embed_size, args.smiles_hidden_dim,
                                                 args.smiles_use_bidirectional, args.smiles_n_layer,
                                                 args.smiles_dropout, args.smiles_use_lstm).to(args.device)
        self.GeoModel = SchNet(args.sch_n_interaction, args.sch_hidden_dim, args.sch_n_gaussian,
                               args.sch_n_filter, args.sch_cutoff).to(args.device)
        if args.rnn_use_lstm:
            self.GeoRNNModel = GeoLSTM(args.sch_hidden_dim, args.rnn_hidden_dim, args.rnn_use_bidirectional,
                                       args.rnn_n_layer, args.rnn_dropout, args.rnn_use_lstm).to(args.device)
        else:
            self.GeoRNNModel = GeoGRU(args.sch_hidden_dim, args.rnn_hidden_dim, args.rnn_use_bidirectional,
                                      args.rnn_n_layer, args.rnn_dropout, args.rnn_use_lstm).to(args.device)

        self._set_optimizer()

    def _set_optimizer(self):
        self.model_param_group = [{'params': self.SMILESRNNModel.parameters(),
                                   'lr': self.args.lr * self.args.smiles_lr_scale},
                                  {'params': self.GeoModel.parameters(),
                                   'lr': self.args.lr * self.args.rnn_lr_scale},
                                  {'params': self.GeoRNNModel.parameters(),
                                   'lr': self.args.lr * self.args.geo_lr_scale},
                                  {'params': self.Projector.parameters(),
                                   'lr': self.args.lr * self.args.projector_lr_scale}]
        self.optimizer = optim.Adam(self.model_param_group, weight_decay=self.args.weight_decay)
        if self.args.lr_type == "step":
            self.scheduler = optim.lr_scheduler. \
                MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        else:
            self.scheduler = optim.lr_scheduler. \
                CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.lr * self.args.cos_rate)

    def training(self, data_loader, epoch, total_step, tag, logger, writer,
                 cl_loss_fc=None, metric="InfoNCE_dot_prod", eval_every=200):
        assert tag in ("training", "validation", "test")
        start_t = time.time()
        if tag == "training":
            self.SMILESRNNModel.train()
            self.GeoModel.train()
            self.GeoRNNModel.train()
            self.Projector.train()
        else:
            self.SMILESRNNModel.eval()
            self.GeoModel.eval()
            self.GeoRNNModel.eval()
            self.Projector.eval()
        loss_accumulation = 0.
        rea_num = 0
        # total, cl, kl
        tmp_loss = torch.zeros((3,))
        for step, (RNN_batch_input, RNN_batch_len, Geo_batch_input, Geo_batch_len) in enumerate(data_loader):
            temp_num = RNN_batch_len.shape[0]
            rea_num += temp_num
            RNN_batch_input, RNN_batch_len, Geo_batch_input, Geo_batch_len \
                = RNN_batch_input.to(self.args.device), RNN_batch_len.to(self.args.device), \
                  Geo_batch_input.to(self.args.device), Geo_batch_len.to(self.args.device)
            if tag == "training":
                RNN_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                Geo_bf = self.GeoModel(Geo_batch_input)
                Geo_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                # projector
                RNN_proj = self.Projector(RNN_bf)
                Geo_proj = self.Projector(Geo_bf)
            else:
                with torch.no_grad():
                    RNN_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                    Geo_bf = self.GeoModel(Geo_batch_input)
                    Geo_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                    # projector
                    RNN_proj = self.Projector(RNN_bf)
                    Geo_proj = self.Projector(Geo_bf)
            cl_loss, kl_loss = cl_loss_fc(RNN_proj, Geo_proj, normalize=True, metric=metric, T=self.args.T,
                                          lambda_cl=self.args.cl_weight, lambda_kl=self.args.kl_weight)
            loss = cl_loss * self.args.cl_weight + kl_loss * self.args.kl_weight
            # backprop
            if tag == "training":
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            loss_item, cl_item, kl_item = loss.item(), cl_loss.item(), kl_loss.item()
            tmp_loss[0] += loss_item
            tmp_loss[1] += cl_item
            tmp_loss[2] += kl_item
            loss_accumulation += (loss_item * temp_num)
            if (step + 1) % eval_every == 0:
                logger.info('Epoch: {:03d} Step: {:06d} total: {:.6f} cl: {:.6f} kl: {:.10f} time: {:.2f}s'
                            .format(epoch, step + 1, tmp_loss[0] / eval_every, tmp_loss[1] / eval_every,
                                    tmp_loss[2] / eval_every, time.time() - start_t))
                tmp_loss = torch.zeros((3,))
            if writer is not None:
                writer.add_scalar("loss/{}".format(tag), loss_item, total_step + step + 1)
                writer.add_scalar("loss/{}_cl".format(tag), cl_item, total_step + step + 1)
                writer.add_scalar("loss/{}_kl".format(tag), kl_item, total_step + step + 1)
        return loss_accumulation / rea_num, time.time() - start_t, total_step + step + 1

    def save(self, epoch, best_p, is_best, model_save_dir):
        save_dict = {"epoch": epoch, "best_performance": best_p, "optimizer": self.optimizer.state_dict(),
                     "smiles_rnn_state_dict": self.SMILESRNNModel.state_dict(),
                     "geo_state_dict": self.GeoModel.state_dict(), "geo_rnn_state_dict": self.GeoRNNModel.state_dict(),
                     "projector_state_dict": self.Projector.state_dict()}
        save_checkpoint(
            save_dict, is_best,
            os.path.join(model_save_dir, "model", "{}.pt".
                         format(self.args.log_path)))


class ReactionSLModel:
    def __init__(self, args):
        super(ReactionSLModel, self).__init__()
        self.args = args
        if args.smiles_use_lstm:
            self.SMILESRNNModel = EmbedLSTMSMILES(args.vocab_size, args.smiles_embed_size, args.smiles_hidden_dim,
                                                  args.smiles_use_bidirectional, args.smiles_n_layer,
                                                  args.smiles_dropout, args.smiles_use_lstm).to(args.device)
        else:
            self.SMILESRNNModel = EmbedGRUSMILES(args.vocab_size, args.smiles_embed_size, args.smiles_hidden_dim,
                                                 args.smiles_use_bidirectional, args.smiles_n_layer,
                                                 args.smiles_dropout, args.smiles_use_lstm).to(args.device)
        self.GeoModel = SchNet(args.sch_n_interaction, args.sch_hidden_dim, args.sch_n_gaussian,
                               args.sch_n_filter, args.sch_cutoff).to(args.device)
        if args.rnn_use_lstm:
            self.GeoRNNModel = GeoLSTM(args.sch_hidden_dim, args.rnn_hidden_dim, args.rnn_use_bidirectional,
                                       args.rnn_n_layer, args.rnn_dropout, args.rnn_use_lstm).to(args.device)
        else:
            self.GeoRNNModel = GeoGRU(args.sch_hidden_dim, args.rnn_hidden_dim, args.rnn_use_bidirectional,
                                      args.rnn_n_layer, args.rnn_dropout, args.rnn_use_lstm).to(args.device)

        self._set_optimizer()

    def _set_optimizer(self):
        multi = 2 if self.args.rnn_use_bidirectional else 1
        cat = 2 if self.args.data_type == "rnn_geo" else 1
        bn = True if self.args.predictor_bn == 1 else False
        self.Predictor = MLP(input_dim=self.args.rnn_hidden_dim * multi * cat,
                             hidden_dim=self.args.rnn_hidden_dim * multi * cat,
                             output_dim=1, num_layers=self.args.predictor_num_layers,
                             dropout=self.args.predictor_dropout, bn=bn).to(self.args.device)
        self.model_param_group = [{'params': self.Predictor.parameters(),
                                   'lr': self.args.lr * self.args.predictor_lr_scale}]
        if self.args.mlp_only == 0:
            if "rnn" in self.args.data_type:
                self.model_param_group.append({'params': self.SMILESRNNModel.parameters(),
                                               'lr': self.args.lr * self.args.smiles_lr_scale})
            if "geo" in self.args.data_type:
                self.model_param_group.extend([{'params': self.GeoModel.parameters(),
                                                'lr': self.args.lr * self.args.rnn_lr_scale},
                                               {'params': self.GeoRNNModel.parameters(),
                                                'lr': self.args.lr * self.args.geo_lr_scale}])

        if self.args.opt == "adam":
            self.optimizer = optim.Adam(self.model_param_group, weight_decay=self.args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model_param_group, lr=self.args.lr,
                                       weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        if self.args.lr_type == "step":
            self.scheduler = optim.lr_scheduler. \
                MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        else:
            self.scheduler = optim.lr_scheduler. \
                CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.lr * self.args.cos_rate)

    def load_state_dict(self, pretraining_state_dict):
        if "rnn" in self.args.data_type:
            self.SMILESRNNModel.load_state_dict(pretraining_state_dict["smiles_rnn_state_dict"])
        if "geo" in self.args.data_type:
            self.GeoModel.load_state_dict(pretraining_state_dict["geo_state_dict"])
            self.GeoRNNModel.load_state_dict(pretraining_state_dict["geo_rnn_state_dict"])
        # if self.args.supervised == 1:
        #     self.Predictor.load_state_dict(pretraining_state_dict["predictor_state_dict"])

    def training(self, data_loader, epoch, total_step, tag, logger, writer,
                 loss_fc=None, mse_fc=None, mae_fc=None, eval_every=1000000):
        assert tag in ("training", "validation", "test")
        start_t = time.time()
        if tag == "training":
            self.Predictor.train()
            # rubbish codes boy
            if self.args.mlp_only == 0:
                self.SMILESRNNModel.train()
                self.GeoModel.train()
                self.GeoRNNModel.train()
            else:
                self.SMILESRNNModel.eval()
                self.GeoModel.eval()
                self.GeoRNNModel.eval()
        else:
            self.Predictor.eval()
            self.SMILESRNNModel.eval()
            self.GeoModel.eval()
            self.GeoRNNModel.eval()
        loss_accumulation = [0., 0., 0.]
        rea_num = 0
        tmp_loss = [[], [], []]
        y, pred = [], []
        for step, (RNN_batch_input, RNN_batch_len, Geo_batch_input, Geo_batch_len, batch_y) in enumerate(data_loader):
            temp_num = RNN_batch_len.shape[0]
            rea_num += temp_num
            RNN_batch_input, RNN_batch_len, Geo_batch_input, Geo_batch_len \
                = RNN_batch_input.to(self.args.device), RNN_batch_len.to(self.args.device), \
                  Geo_batch_input.to(self.args.device), Geo_batch_len.to(self.args.device)
            if tag == "training":
                if self.args.data_type == "rnn_geo":
                    RNN_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                    Geo_bf = self.GeoModel(Geo_batch_input)
                    Geo_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                    # predictor
                    cat_bf = torch.concat([RNN_bf, Geo_bf], dim=1)
                elif self.args.data_type == "rnn":
                    cat_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                else:
                    Geo_bf = self.GeoModel(Geo_batch_input)
                    cat_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                out = self.Predictor(cat_bf)
            else:
                with torch.no_grad():
                    if self.args.data_type == "rnn_geo":
                        RNN_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                        Geo_bf = self.GeoModel(Geo_batch_input)
                        Geo_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                        # predictor
                        cat_bf = torch.concat([RNN_bf, Geo_bf], dim=1)
                    elif self.args.data_type == "rnn":
                        cat_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                    else:
                        Geo_bf = self.GeoModel(Geo_batch_input)
                        cat_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                    out = self.Predictor(cat_bf)
            y.append(batch_y)
            pred.append(out.detach().cpu() * self.args.training_std + self.args.training_mean)
            batch_y = (batch_y - self.args.training_mean) / self.args.training_std
            batch_y = batch_y.reshape(out.shape).to(self.args.device)
            loss = loss_fc(out, batch_y)
            mse = mse_fc(out, batch_y)
            mae = mae_fc(out, batch_y)
            # backprop
            if tag == "training":
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            loss_item = loss.item()
            mse_item = mse.item()
            mae_item = mae.item()
            tmp_loss[0].append(loss_item)
            tmp_loss[1].append(mse_item)
            tmp_loss[2].append(mae_item)
            loss_accumulation[0] += (loss_item * temp_num)
            loss_accumulation[1] += (mse_item * temp_num)
            loss_accumulation[2] += (mae_item * temp_num)

            if (step + 1) % eval_every == 0:
                logger.info('Epoch: {:03d} Step: {:06d} mse: {:.6f} mae: {:.6f} time: {:.2f}s'
                            .format(epoch, step + 1, np.mean(tmp_loss[1]), np.mean(tmp_loss[2]), time.time() - start_t))
                tmp_loss = [[], [], []]
            if writer is not None:
                writer.add_scalar("loss/{}".format(tag), loss_item, total_step + step + 1)
        y, pred = torch.concat(y), torch.concat(pred)
        epoch_loss = [np.sqrt(loss_accumulation[1] / rea_num) * self.args.training_std,
                      loss_accumulation[2] / rea_num * self.args.training_std,
                      r2_score(y, pred)]
        return epoch_loss, time.time() - start_t, total_step + step + 1, y, pred

    def save(self, epoch, best_p, is_best, model_save_dir):
        save_dict = {"epoch": epoch, "best_performance": best_p, "optimizer": self.optimizer.state_dict(),
                     "mean": self.args.training_mean, "std": self.args.training_std,
                     "smiles_rnn_state_dict": self.SMILESRNNModel.state_dict(),
                     "geo_state_dict": self.GeoModel.state_dict(), "geo_rnn_state_dict": self.GeoRNNModel.state_dict(),
                     "predictor_state_dict": self.Predictor.state_dict()}
        save_checkpoint(
            save_dict, is_best,
            os.path.join(model_save_dir, "model", "{}_{}.pt".
                         format(self.args.log_path, self.args.supervised)))


# the same as ReactionSLModel(ReactionCLModel)?
class FinetuneModel(ReactionSLModel):
    def __init__(self, args):
        super(FinetuneModel, self).__init__(args=args)

    def _set_optimizer(self):
        multi = 2 if self.args.rnn_use_bidirectional else 1
        cat = 2 if self.args.data_type == "rnn_geo" else 1
        bn = True if self.args.predictor_bn == 1 else False
        self.Predictor = MLP(input_dim=self.args.rnn_hidden_dim * multi * cat,
                             hidden_dim=self.args.rnn_hidden_dim * multi * cat,
                             output_dim=1, num_layers=self.args.predictor_num_layers,
                             dropout=self.args.predictor_dropout, bn=bn).to(self.args.device)
        self.model_param_group = [{'params': self.Predictor.parameters(),
                                   'lr': self.args.lr * self.args.predictor_lr_scale}]
        if self.args.mlp_only == 0:
            if "rnn" in self.args.data_type:
                self.model_param_group.append({'params': self.SMILESRNNModel.parameters(),
                                               'lr': self.args.lr * self.args.smiles_lr_scale})
            if "geo" in self.args.data_type:
                self.model_param_group.extend([{'params': self.GeoModel.parameters(),
                                                'lr': self.args.lr * self.args.rnn_lr_scale},
                                               {'params': self.GeoRNNModel.parameters(),
                                                'lr': self.args.lr * self.args.geo_lr_scale}])

        if self.args.opt == "adam":
            self.optimizer = optim.Adam(self.model_param_group, weight_decay=self.args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model_param_group, lr=self.args.lr,
                                       weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        if self.args.lr_type == "step":
            self.scheduler = optim.lr_scheduler. \
                MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        else:
            self.scheduler = optim.lr_scheduler. \
                CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.lr * self.args.cos_rate)

    def load_state_dict(self, pretraining_state_dict):
        if "rnn" in self.args.data_type:
            self.SMILESRNNModel.load_state_dict(pretraining_state_dict["smiles_rnn_state_dict"])
        if "geo" in self.args.data_type:
            self.GeoModel.load_state_dict(pretraining_state_dict["geo_state_dict"])
            self.GeoRNNModel.load_state_dict(pretraining_state_dict["geo_rnn_state_dict"])
        if self.args.supervised in [1, 2]:
            self.Predictor.load_state_dict(pretraining_state_dict["predictor_state_dict"])

    def training(self, data_loader, epoch, total_step, tag, logger, writer,
                 loss_fc=None, mse_fc=None, mae_fc=None, eval_every=1000000):
        assert tag in ("training", "validation", "test")
        start_t = time.time()
        if tag == "training":
            self.Predictor.train()
            if self.args.mlp_only == 0:
                self.SMILESRNNModel.train()
                self.GeoModel.train()
                self.GeoRNNModel.train()
        else:
            self.Predictor.eval()
            self.SMILESRNNModel.eval()
            self.GeoModel.eval()
            self.GeoRNNModel.eval()
        loss_accumulation = [0., 0., 0.]
        rea_num = 0
        tmp_loss = [[], [], []]
        y, pred = [], []
        for step, (RNN_batch_input, RNN_batch_len, Geo_batch_input, Geo_batch_len, batch_y) in enumerate(data_loader):
            temp_num = RNN_batch_len.shape[0]
            rea_num += temp_num
            RNN_batch_input, RNN_batch_len, Geo_batch_input, Geo_batch_len \
                = RNN_batch_input.to(self.args.device), RNN_batch_len.to(self.args.device), \
                  Geo_batch_input.to(self.args.device), Geo_batch_len.to(self.args.device)
            if tag == "training":
                if self.args.mlp_only == 0:
                    if self.args.data_type == "rnn_geo":
                        RNN_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                        Geo_bf = self.GeoModel(Geo_batch_input)
                        Geo_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                        # predictor
                        cat_bf = torch.concat([RNN_bf, Geo_bf], dim=1)
                    elif self.args.data_type == "rnn":
                        cat_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                    else:
                        Geo_bf = self.GeoModel(Geo_batch_input)
                        cat_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                else:
                    with torch.no_grad():
                        if self.args.data_type == "rnn_geo":
                            RNN_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                            Geo_bf = self.GeoModel(Geo_batch_input)
                            Geo_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                            # predictor
                            cat_bf = torch.concat([RNN_bf, Geo_bf], dim=1)
                        elif self.args.data_type == "rnn":
                            cat_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                        else:
                            Geo_bf = self.GeoModel(Geo_batch_input)
                            cat_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                out = self.Predictor(cat_bf)
            else:
                with torch.no_grad():
                    if self.args.data_type == "rnn_geo":
                        RNN_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                        Geo_bf = self.GeoModel(Geo_batch_input)
                        Geo_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                        # predictor
                        cat_bf = torch.concat([RNN_bf, Geo_bf], dim=1)
                    elif self.args.data_type == "rnn":
                        cat_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
                    else:
                        Geo_bf = self.GeoModel(Geo_batch_input)
                        cat_bf = self.GeoRNNModel(Geo_bf, Geo_batch_len)
                    out = self.Predictor(cat_bf)
            y.append(batch_y)
            pred.append(out.detach().cpu() * self.args.training_std + self.args.training_mean)
            batch_y = (batch_y - self.args.training_mean) / self.args.training_std
            batch_y = batch_y.reshape(out.shape).to(self.args.device)
            loss = loss_fc(out, batch_y)
            mse = mse_fc(out, batch_y)
            mae = mae_fc(out, batch_y)
            # backprop
            if tag == "training":
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            loss_item = loss.item()
            mse_item = mse.item()
            mae_item = mae.item()
            tmp_loss[0].append(loss_item)
            tmp_loss[1].append(mse_item)
            tmp_loss[2].append(mae_item)
            loss_accumulation[0] += (loss_item * temp_num)
            loss_accumulation[1] += (mse_item * temp_num)
            loss_accumulation[2] += (mae_item * temp_num)

            if (step + 1) % eval_every == 0:
                logger.info('Epoch: {:03d} Step: {:06d} mse: {:.6f} mae: {:.6f} time: {:.2f}s'
                            .format(epoch, step + 1, np.mean(tmp_loss[1]), np.mean(tmp_loss[2]), time.time() - start_t))
                tmp_loss = [[], [], []]
            if writer is not None:
                writer.add_scalar("loss/{}".format(tag), loss_item, total_step + step + 1)
        y, pred = torch.concat(y), torch.concat(pred)
        pred = torch.clamp(pred, 0., 100.)
        epoch_loss = [loss_fc(y, pred).item(),
                      np.sqrt(mse_fc(y, pred).item()),
                      mae_fc(y, pred).item(),
                      r2_score(y, pred)]
        return epoch_loss, time.time() - start_t, total_step + step + 1, y, pred

    # just test SHAP
    def __call__(self, data, total_step=0, tag="test",
                 loss_fc=None, mse_fc=None, mae_fc=None, **kwargs):
        start_t = time.time()
        if tag == "training":
            self.Predictor.train()
            if self.args.mlp_only == 0:
                self.SMILESRNNModel.train()
                self.GeoModel.train()
                self.GeoRNNModel.train()
        else:
            self.Predictor.eval()
            self.SMILESRNNModel.eval()
            self.GeoModel.eval()
            self.GeoRNNModel.eval()
        loss_accumulation = [0., 0., 0.]
        rea_num = 0
        tmp_loss = [[], [], []]
        y, pred = [], []

        RNN_batch_input = rnn_utils.pad_sequence([data], batch_first=True, padding_value=0).to(self.args.device)
        RNN_batch_len = torch.tensor([data.shape[0]]).to(self.args.device)
        temp_num = 1
        rea_num += temp_num
        with torch.no_grad():
            cat_bf = self.SMILESRNNModel(RNN_batch_input, RNN_batch_len)
            out = self.Predictor(cat_bf)
            pred.append(out.detach().cpu() * self.args.training_std + self.args.training_mean)
            # batch_y = (batch_y - self.args.training_mean) / self.args.training_std
            # batch_y = batch_y.reshape(out.shape).to(self.args.device)
            # loss = loss_fc(out, batch_y)
            # mse = mse_fc(out, batch_y)
            # mae = mae_fc(out, batch_y)
            # # backprop
            # if tag == "training":
            #     self.optimizer.zero_grad(set_to_none=True)
            #     loss.backward()
            #     self.optimizer.step()
            # loss_item = loss.item()
            # mse_item = mse.item()
            # mae_item = mae.item()
            # tmp_loss[0].append(loss_item)
            # tmp_loss[1].append(mse_item)
            # tmp_loss[2].append(mae_item)
            # loss_accumulation[0] += (loss_item * temp_num)
            # loss_accumulation[1] += (mse_item * temp_num)
            # loss_accumulation[2] += (mae_item * temp_num)
        # epoch_loss = [np.sqrt(loss_accumulation[1] / rea_num) * self.args.training_std,
        #               loss_accumulation[2] / rea_num * self.args.training_std,
        #               r2_score(torch.concat(y), torch.concat(pred))]
        return torch.tensor(pred)

    def save(self, epoch, best_p, is_best, model_save_dir):
        save_dict = {"epoch": epoch, "best_performance": best_p, "optimizer": self.optimizer.state_dict(),
                     "mean": self.args.training_mean, "std": self.args.training_std,
                     "smiles_rnn_state_dict": self.SMILESRNNModel.state_dict(),
                     "geo_state_dict": self.GeoModel.state_dict(), "geo_rnn_state_dict": self.GeoRNNModel.state_dict(),
                     "predictor_state_dict": self.Predictor.state_dict()}
        save_checkpoint(
            save_dict, is_best,
            os.path.join(model_save_dir, "model", "{}_{}.pt".
                         format(self.args.log_path, self.args.supervised)))
