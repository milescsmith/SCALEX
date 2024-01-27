#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Mon 18 Nov 2019 01:16:06 PM CST

# File Name: vae.py
# Description:

"""

from collections import defaultdict
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from tqdm.autonotebook import tqdm, trange

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.nn.functional import binary_cross_entropy  # , cosine_similarity
from tqdm.auto import tqdm, trange

from scalex.net.layer import NN, Encoder
from scalex.net.loss import kl_div
from scalex.net.layer import NN, Encoder
from scalex.net.loss import kl_div


class VAE(nn.Module):
    """
    VAE framework
    """

    def __init__(self, enc, dec, n_domain=1):
        """
        Parameters
        ----------
        enc
            Encoder structure config
        dec
            Decoder structure config
        n_domain
            The number of different domains
        """
        super().__init__()
        x_dim = dec[-1][1]
        z_dim = enc[-1][1]
        self.encoder = Encoder(x_dim, enc)
        self.decoder = NN(z_dim, dec)
        self.n_domain = n_domain
        self.x_dim = x_dim
        self.z_dim = z_dim

    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)  # noqa: ARG005
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)  # noqa: ARG005
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def encodeBatch(
        self,
        dataloader,
        device="cuda",
        out="latent",
        batch_id=None,
        return_idx=False,
        evaluate=False,
    ):
        """
        Inference

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for inference.
        device
            'cuda' or 'cpu' for . Default: 'cuda'.
        out
            The inference layer for output. If 'latent', output latent feature z. If 'impute', output imputed gene expression matrix. Default: 'latent'.
        batch_id
            If None, use batch 0 decoder to infer for all samples. Else, use the corresponding decoder according to the sample batch id to infer for each sample.
        return_idx
            Whether return the dataloader sample index. Default: False.
        eval
            If True, set the model to evaluation mode. If False, set the model to train mode. Default: False.

        Returns
        -------
        Inference layer and sample index (if return_idx=True).
        """
        self.to(device)
        if evaluate:
        if evaluate:
            self.eval()
        else:
            self.train()
        indices = np.zeros(dataloader.dataset.shape[0])
        if out == "latent":
            output = np.zeros((dataloader.dataset.shape[0], self.z_dim))

            for x, _, idx in dataloader:
                _x = x.float().to(device)
                z = self.encoder(_x)[1]  # z, mu, var
                output[idx] = z.detach().cpu().numpy()
                indices[idx] = idx
        elif out == "impute":
            output = np.zeros((dataloader.dataset.shape[0], self.x_dim))

            if batch_id in dataloader.dataset.adata.obs["batch"].cat.categories:
                batch_id = list(dataloader.dataset.adata.obs["batch"].cat.categories).index(batch_id)
            else:
                batch_id = 0

            for x, _, idx in dataloader:
                _x = x.float().to(device)
                z = self.encoder(_x)[1]  # z, mu, var
                output[idx] = self.decoder(z, torch.LongTensor([batch_id] * len(z))).detach().cpu().numpy()
                indices[idx] = idx

        return (output, indices) if return_idx else output
        return (output, indices) if return_idx else output

    def fit(
        self,
        dataloader,
        lr=2e-4,
        max_iteration=30000,
        early_stopping=None,
        device="cuda",
        verbose=False,
    ):
        """
        Fit model

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        lr
            Learning rate. Default: 2e-4.
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None.
        device
            'cuda' or 'cpu' for training. Default: 'cuda'.
        verbose
            Verbosity, True or False. Default: False.
        """
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        num_epoch = int(np.ceil(max_iteration / len(dataloader)))

        with trange(n_epoch, total=n_epoch, desc="Epochs") as tq:
            for _epoch in tq:
                epoch_loss = defaultdict(float)
                i = 0
                for x, y, _idx in (
                    tk0 := tqdm(
                        dataloader, total=len(dataloader), leave=False, desc="Iterations", disable=(not verbose)
                    )
                ):
                    _x, _y = x.float().to(device), y.long().to(device)

                    # loss
                    z, mu, var = self.encoder(x)
                    recon_x = self.decoder(z, y)
                    recon_loss = binary_cross_entropy(recon_x, x) * x.size(
                        -1
                    )  ## TO DO
                    kl_loss = kl_div(mu, var)

                    loss = {"recon_loss": recon_loss, "kl_loss": 0.5 * kl_loss}

                    logger.debug(f"Resetting gradients for iteration {i} of epoch {num_epoch}")
                    optim.zero_grad()
                    sum(loss.values()).backward()
                    optim.step()

                    for k in loss:
                    for k in loss:
                        epoch_loss[k] += loss[k].item()

                    info = ",".join([f"{k}={v:.3f}" for k, v in loss.items()])
                    info = ",".join([f"{k}={v:.3f}" for k, v in loss.items()])
                    tk0.set_postfix_str(info)
                    i += 1

                epoch_loss = {k: v / (i + 1) for k, v in epoch_loss.items()}
                epoch_info = ",".join(
                    [f"{k}={v:.3f}" for k, v in epoch_loss.items()]
                )
                tq.set_postfix_str(epoch_info)

                early_stopping(sum(epoch_loss.values()), self)
                if early_stopping.early_stop:
                    logger.info(f"EarlyStopping: run {_epoch + 1} epoch")
                    break


# this isn't used, but keeping it to better match upstream
# def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
#     x_centered = x - x.mean(dim = dim, keepdim = True)
#     y_centered = y - y.mean(dim = dim, keepdim = True)
#     return cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)
