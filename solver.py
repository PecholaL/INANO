""" solver for training Generator Gano
"""

import torch
import numpy
import yaml
from nflows.distributions import normal
from Gano import getGLOW
from data.dataloader import SpeakerDataset, get_dataloader, infinite_iter

gano = getGLOW()


class Solver(object):
    def __init__(self, trainConfig_path, args):
        with open(trainConfig_path, "r") as f:
            self.trainConfig = yaml.safe_load(f)
        self.args = args
        self.get_inf_train_iter()
        self.build_model()
        self.build_optim()

        if self.args.load_model:
            self.load_model()

    # prepare data, called in Solver.init
    def get_inf_train_iter(self):
        spkDs = SpeakerDataset(self.args.dataset_path)
        self.train_iter = infinite_iter(
            get_dataloader(
                dataset=spkDs,
                batch_size=self.trainConfig['data_loader']['batch_size'],
                num_workers=self.trainConfig['data_loader']['num_workers'],
            )
        )
        print("[INANO]infinite dataloader built")
        return

    # load model/data to cuda
    def cc(self, net):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return net.to(device)

    # called in Solver.init
    def build_model(self):
        self.gano = self.cc(getGLOW())
        print("[INANO]gano built")
        print(
            "[INANO]gano parameter count: {}".format(
                sum(x.numel() for x in self.gano.parameters())
            )
        )

    # called in Solver.init
    def build_optim(self):
        params = list(
            filter(lambda p: p.requires_grad, self.gano.parameters())
        )

        lr = self.trainConfig["optimizer"]["lr"]
        beta1 = self.trainConfig["optimizer"]["beta1"]
        beta2 = self.trainConfig["optimizer"]["beta2"]
        eps = eval(self.trainConfig["optimizer"]["eps"])
        weight_decay = self.trainConfig["optimizer"]["weight_decay"]

        self.optim = torch.optim.Adam(
            params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        print("[INANO]optimizer built")

    # autosave, called in training
    def save_model(self):
        torch.save(
            self.gano.state_dict(),
            f"{self.args.store_model_path}gano.ckpt",
        )
        torch.save(
            self.optim.state_dict(),
            f"{self.args.store_model_path}optim.opt",
        )

    # load trained model
    def load_model(self):
        print(f"[INANO]load gano {self.args.load_model_path}")
        self.model.load_state_dict(torch.load(f"{self.args.load_model_path}gano.ckpt"))
        self.optim.load_state_dict(
            torch.load(f"{self.args.load_model_path}optim.opt")
        )
        return

    # training
    def train(self, n_iterations):
        print("[INANO]starting training...")
        loss_history = []
        for iter in range(n_iterations):
            # get data for current iteration
            spk_emb = next(self.train_iter).reshape(-1,1,8,8).to(torch.float32)

            ## load to cuda
            spk_emb = self.cc(spk_emb)

            output, logabsdet = self.gano(spk_emb)
            shape = output.shape[1:]
            log_z = normal.StandardNormal(shape=shape).log_prob(output)
            loss = log_z + logabsdet
            loss = -loss.mean()/(8 * 8 * 1)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            loss_history.append(loss.item())

            # logging
            print(
                f"[INANO]:[{iter+1}/{n_iterations}]",
                f"loss={loss.item():6f}",
                end="\r",
            )

            # summary
            if (iter + 1) % self.args.summary_steps == 0 or iter + 1 == n_iterations:
                print()

            # autosave
            if (iter + 1) % self.args.save_steps == 0 or iter + 1 == n_iterations:
                self.save_model()
        numpy.save(f"{self.args.store_model_path}loss_history.npy", numpy.array(loss_history))
        return