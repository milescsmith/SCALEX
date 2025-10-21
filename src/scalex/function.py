#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Tue 29 Sep 2020 01:41:23 PM CST

# File Name: function.py
# Description:

"""

from anndata import concat
# from typing import Union, List


def SCALEX(
    data_list,  # Union[str, AnnData, List]=None,
    batch_categories: list = None,
    profile: str = "RNA",
    batch_name: str = "batch",
    min_features: int = 600,
    min_cells: int = 3,
    target_sum: int = None,
    n_top_features: int = None,
    min_cell_per_batch: int = 200,
    join: str = "inner",
    batch_key: str = "batch",
    processed: bool = False,
    fraction: float = None,
    n_obs: int = None,
    use_layer: str = "X",
    keep_mt: bool = False,
    backed: bool = False,
    batch_size: int = 64,
    groupby: str = None,
    subsets: list = None,
    lr: float = 2e-4,
    max_iteration: int = 30000,
    seed: int = 124,
    gpu: int = 0,
    outdir: str = None,
    projection: str = None,
    repeat: bool = False,
    name: str = None,
    impute: str = None,
    chunk_size: int = 20000,
    ignore_umap: bool = False,
    verbose: bool = False,
    assess: bool = False,
    show: bool = True,
    eval: bool = False,
    num_workers: int = 4,
    cell_type: str = "cell_type",
):  # -> AnnData:
    """
    Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space

    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    profile
        Specify the single-cell profile, RNA or ATAC. Default: RNA.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    min_features
        Filtered out cells that are detected in less than min_features. Default: 600.
    min_cells
        Filtered out genes that are detected in less than min_cells. Default: 3.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches.
    batch_key
        Add the batch annotation to obs using this key. By default, batch_key='batch'.
    batch_size
        Number of samples per batch to load. Default: 64.
    lr
        Learning rate. Default: 2e-4.
    max_iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
    seed
        Random seed for torch and numpy. Default: 124.
    gpu
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: 'output/'.
    projection
        Use for new dataset projection. Input the folder containing the pre-trained model. If None, don't do projection. Default: None.
    repeat
        Use with projection. If False, concatenate the reference and projection datasets for downstream analysis. If True, only use projection datasets. Default: False.
    impute
        If True, calculate the imputed gene expression and store it at adata.layers['impute']. Default: False.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    ignore_umap
        If True, do not perform UMAP for visualization and leiden for clustering. Default: False.
    verbose
        Verbosity, True or False. Default: False.
    assess
        If True, calculate the entropy_batch_mixing score and silhouette score to evaluate integration results. Default: False.

    Returns
    -------
    The output folder contains:
    adata.h5ad
        The AnnData matrice after batch effects removal. The low-dimensional representation of the data is stored at adata.obsm['latent'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    log.txt
        Records raw data information, filter conditions, model parameters etc.
    umap.pdf
        UMAP plot for visualization.
    """
    import torch
    import numpy as np
    import pandas as pd
    import os
    import scanpy as sc

    from .data import load_data
    from .net.vae import VAE
    from .net.utils import EarlyStopping
    from .metrics import batch_entropy_mixing_score, silhouette_score
    from .logger import create_logger
    from .plot import embedding

    np.random.seed(seed)  # seed
    torch.manual_seed(seed)

    if torch.cuda.is_available():  # cuda device
        device = "cuda"
        torch.cuda.set_device(gpu)
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch.set_default_device(device)

    if outdir:
        if name is not None and projection is not None:
            outdir = os.path.join(projection, "projection", name)
            os.makedirs(outdir, exist_ok=True)
        # outdir = outdir+'/'
        outdir.joinpath("checkpoint").mkdir(exist_ok=True, parents=True)
        log = create_logger("SCALEX", fh=outdir.joinpath("log.txt"), overwrite=True)
    else:
        log = create_logger("SCALEX")
    if projection:
        projection = projection if isinstance(projection, Path) else Path(projection)
    data_list = data_list if isinstance(data_list, list) else [data_list]
    if raw_data_list:
        raw_data_list = raw_data_list if isinstance(raw_data_list, list) else [raw_data_list]

    if not projection:
        adata, trainloader, testloader = load_data(
            data_list,
            raw_data_list,
            batch_categories,
            join=join,
            profile=profile,
            target_sum=target_sum,
            n_top_features=n_top_features,
            min_cell_per_batch=min_cell_per_batch,
            batch_size=batch_size,
            groupby=groupby,
            subsets=subsets,
            chunk_size=chunk_size,
            min_features=min_features,
            min_cells=min_cells,
            fraction=fraction,
            n_obs=n_obs,
            processed=processed,
            use_layer=use_layer,
            backed=backed,
            batch_name=batch_name,
            batch_key=batch_key,
            keep_mt=keep_mt,
            log=log,
            num_workers=num_workers,
        )

        # TODO: if the model exists, why not just reload it?
        early_stopping = EarlyStopping(
            patience=10, checkpoint_file=str(outdir.joinpath("checkpoint", "model.pt")) if outdir else "tmp_model.pt"
        )
        x_dim = adata.shape[1] if use_layer is None else adata.obsm[use_layer].shape[1]
        n_domain = len(adata.obs["batch"].cat.categories)

        # model config
        enc = [["fc", 1024, 1, "relu"], ["fc", 10, "", ""]]  # TO DO
        dec = [["fc", x_dim, n_domain, "sigmoid"]]

        model = VAE(enc, dec, n_domain=n_domain)

        model.fit(
            trainloader,
            lr=lr,
            max_iteration=max_iteration,
            device=device,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        if outdir:
            config_file = outdir.joinpath("checkpoint/config.pt")
            torch.save({"n_top_features": adata.var.index, "enc": enc, "dec": dec, "n_domain": n_domain}, config_file)
    else:
        state = torch.load(os.path.join(projection, "checkpoint/config.pt"), weights_only=False)
        n_top_features, enc, dec, n_domain = state["n_top_features"], state["enc"], state["dec"], state["n_domain"]
        model = VAE(enc, dec, n_domain=n_domain)
        model.load_model(projection.joinpath("checkpoint", "model.pt"))
        model.to(device)

        adata, trainloader, testloader = load_data(
            data_list,
            batch_categories,
            join="outer",
            profile=profile,
            target_sum=target_sum,
            n_top_features=n_top_features,
            min_cells=0,
            min_features=min_features,
            min_cell_per_batch=min_cell_per_batch,
            processed=processed,
            batch_name=batch_name,
            batch_key=batch_key,
            # keep_mt=keep_mt,
            log=log,
            num_workers=num_workers,
        )

    adata.obsm["latent"] = model.encodeBatch(testloader, device=device, evaluate=evaluate)  # save latent rep
    if impute:
        adata.layers["impute"] = model.encodeBatch(
            testloader, out="impute", batch_id=impute, device=device, evaluate=evaluate
        )

    model.to(device)
    del model
    if projection and (not repeat):
        ref = sc.read_h5ad(os.path.join(projection, "adata.h5ad"))
        adata = concat(
            [ref, adata],
            # join='outer',
            label="projection",
            keys=["reference", "query"],
            index_unique=None,
        )
        # adata.raw = concat([ref.raw.to_adata(), adata.raw.to_adata()], join='outer', label='projection', keys=['reference', 'query'])
        if "leiden" in adata.obs:
            del adata.obs["leiden"]
        for col in adata.obs.columns:
            if not pd.api.types.is_string_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)

    # if outdir is not None:
    #     adata.write(os.path.join(outdir, 'adata.h5ad'), compression='gzip')

    if not ignore_umap:  # and adata.shape[0]<1e6:
        logger.info("Plot umap")
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="latent")
        sc.tl.umap(adata, min_dist=0.1)
        sc.tl.leiden(adata)
        adata.obsm["X_scalex_umap"] = adata.obsm["X_umap"]

        # UMAP visualization
        sc.set_figure_params(dpi=80, figsize=(3, 3))
        cols = [cell_type, "leiden"]
        cols += ["batch"] if n_domain > 1 else []
        color = [c for c in cols if c in adata.obs]
        if outdir:
            sc.settings.figdir = outdir
            save = ".png"
        else:
            save = None

        if color:
            if projection and (not repeat):
                cell_type = cell_type if cell_type in adata.obs else "leiden"
                save = os.path.join(outdir, "umap.png")
                embedding(adata, color=cell_type, groupby="projection", save=save, show=show)
            else:
                sc.pl.umap(adata, color=color, save=save, wspace=0.4, ncols=4, show=show)

    if outdir is not None:
        adata.write(os.path.join(outdir, "adata.h5ad"), compression="gzip")

    if assess:
        if adata.shape[0] > 5e4:
            log.info("Subsample cell numbers to 50,000 for entropy_batch_mixing_score calculation.")
            sc.pp.subsample(adata, n_obs=int(5e4))
        if len(adata.obs["batch"].cat.categories) > 1:
            entropy_score = batch_entropy_mixing_score(adata.obsm["X_umap"], adata.obs["batch"])
            log.info("batch_entropy_mixing_score: {:.3f}".format(entropy_score))
            adata.uns["batch_entropy_mixing_score"] = entropy_score

        if cell_type in adata.obs:
            sil_score = silhouette_score(adata.obsm["X_umap"], adata.obs[cell_type].cat.codes)
            log.info("silhouette_score: {:.3f}".format(sil_score))
            adata.uns["silhouette_score"] = sil_score

    return adata


def label_transfer(ref, query, rep="latent", label="celltype"):
    """
    Label transfer

    Parameters
    -----------
    ref
        reference containing the projected representations and labels
    query
        query data to transfer label
    rep
        representations to train the classifier. Default is `latent`
    label
        label name. Defautl is `celltype` stored in ref.obs

    Returns
    --------
    transfered label
    """

    from sklearn.neighbors import KNeighborsClassifier

    x_train = ref.obsm[rep]
    y_train = ref.obs[label]
    x_test = query.obsm[rep]

    knn = KNeighborsClassifier().fit(x_train, y_train)
    return knn.predict(x_test)
