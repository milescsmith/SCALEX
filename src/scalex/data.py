#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Wed 26 Dec 2018 03:46:19 PM CST
# File Name: data.py
# Description:
"""

from glob import glob
from pathlib import Path
from typing import Final, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from loguru import logger
from muon import prot as pt
from scipy.sparse import csr, issparse
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from torch import Generator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

CHUNK_SIZE: Final[int] = 20000


def read_mtx(path):
    """\
    Read mtx format data folder including:

        * matrix file: e.g. count.mtx or matrix.mtx or their gz format
        * barcode file: e.g. barcode.txt
        * feature file: e.g. feature.txt

    Parameters
    ----------
    path
        the path store the mtx files

    Return
    ------
    AnnData
    """
    for filename in glob(path + "/*"):
        if ("count" in filename or "matrix" in filename or "data" in filename) and ("mtx" in filename):
            adata = sc.read_mtx(filename).T
    for filename in glob(path + "/*"):
        if "barcode" in filename:
            barcode = pd.read_csv(filename, sep="\t", header=None).iloc[:, -1].values
            adata.obs = pd.DataFrame(index=barcode)
        if "gene" in filename or "peaks" in filename:
            gene = pd.read_csv(filename, sep="\t", header=None).iloc[:, -1].values
            adata.var = pd.DataFrame(index=gene)
        elif "feature" in filename:
            gene = pd.read_csv(filename, sep="\t", header=None).iloc[:, 1].values
            adata.var = pd.DataFrame(index=gene)
    return adata


def load_file(path: str | Path) -> ad.AnnData:
    """
    Load single cell dataset from file

    Parameters
    ----------
    path
        the path store the file

    Return
    ------
    AnnData
    """
    data_path = Path(path)
    adata_file = Path(f"{path}.h5ad")
    if adata_file.exists():
        adata = sc.read(adata_file)
    elif data_path.is_dir():  # mtx format
        adata = read_mtx(data_path)
    elif data_path.is_file():
        if data_path.suffix in {".csv", ".csv.gz"}:
            adata = sc.read_csv(data_path).T
        elif data_path.suffix in {".txt", ".txt.gz", ".tsv", ".tsv.gz"}:
            df = pd.read_csv(data_path, sep="\t", index_col=0).T
            adata = ad.AnnData(df.values, {"obs_names": df.index.values}, {"var_names": df.columns.values})
        elif data_path.suffix == ".h5ad":
            adata = sc.read(data_path)
    elif data_path.suffix in {".h5mu/rna", ".h5mu/atac"}:
        import muon as mu

        adata = mu.read(data_path)
    else:
        msg = f"File {data_path} does not exist"
        raise ValueError(msg)

    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata


def load_files(root) -> ad.AnnData:
    """
    Load single cell dataset from files

    Parameters
    ----------
    root
        the root store the single-cell data files, each file represent one dataset

    Return
    ------
    AnnData
    """
    if root.split("/")[-1] != "*":
        return load_file(root)
    adatas = [load_file(_) for _ in sorted(glob(root))]
    return ad.concat(adatas, label="sub_batch", index_unique=None)


def concat_data(data_list, batch_categories=None, join="inner", batch_key="batch", index_unique=None, save=None):
    """
    Concatenate multiple datasets along the observations axis with name ``batch_key``.

    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a “batch”.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_key
        Add the batch annotation to obs using this key. Default: 'batch'.
    index_unique
        Make the index unique by joining the existing index names with the batch category, using index_unique='-', for instance. Provide None to keep existing indices.
    save
        Path to save the new merged AnnData. Default: None.

    Returns
    -------
    New merged AnnData.
    """
    match data_list:
        case x if len(x) == 1:
            return load_files(data_list[0])
        case x if isinstance(x, str):
            return load_files(data_list)
        case x if isinstance(x, ad.AnnData):
            return data_list
        case _:
            adata_list = [root if isinstance(root, ad.AnnData) else load_files(root) for root in data_list]

            if batch_categories is None:
                batch_categories = list(map(str, range(len(adata_list))))
            elif len(adata_list) != len(batch_categories):
                msg = "The number of adatas to use does not match the number of batches"
                raise ValueError(msg)
            # [print(b, adata.shape) for adata,b in zip(adata_list, batch_categories)]
            concat_adatas = ad.concat(
                adata_list, join=join, label=batch_key, keys=batch_categories, index_unique=index_unique
            )
            if save:
                concat_adatas.write(save, compression="gzip")
            return concat_adatas


def preprocessing(
    adata: ad.AnnData,
    raw_adata: ad.AnnData | None = None,
    profile: Literal["RNA", "ATAC", "PROT"] = "RNA",
    min_features: int = 600,
    min_cells: int = 3,
    target_sum: int | None = None,
    n_top_features=None,  # or gene list
    backed: bool = False,
    # chunk_size: int = CHUNK_SIZE,
) -> ad.AnnData:
    """
    Preprocessing single-cell data

    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
    profile
        Specify the single-cell profile type, RNA or ATAC, Default: RNA.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 100.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. Default 1e4.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.

    Return
    -------
    The AnnData object after preprocessing.

    """
    if target_sum is None:
        target_sum = 10000

    match profile:
        case "RNA":
            return preprocessing_rna(
                adata=adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                n_top_features=n_top_features,
                backed=backed,
            )
        case "ATAC":
            return preprocessing_atac(
                adata=adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                n_top_features=n_top_features,
                backed=backed,
            )
        case "PROT":
            return preprocessing_prot(adata=adata, raw_adata=raw_adata)


def preprocessing_rna(
    adata: ad.AnnData,
    min_features: int = 600,
    min_cells: int = 3,
    target_sum: int = 10000,
    n_top_features: int = 2000,  # or gene list
    backed: bool = False,
) -> ad.AnnData:
    """
    Preprocessing single-cell RNA-seq data

    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 600.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.

    Return
    -------
    The AnnData object after preprocessing.
    """

    logger.info("Preprocessing")
    # if not issparse(adata.X):
    if not isinstance(adata.X, csr.csr_matrix) and (not backed) and (not adata.isbacked):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(("ERCC", "MT-", "mt-"))]]

    logger.info("Filtering cells")
    sc.pp.filter_cells(adata, min_genes=min_features)

    logger.info("Filtering features")
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # TODO: allow one to indicate the type of data and then use the appropriate normalization
    logger.info("Normalizing total per cell")
    sc.pp.normalize_total(adata, target_sum=target_sum)

    logger.info("Log1p transforming")
    sc.pp.log1p(adata)

    adata.raw = adata
    logger.info("Finding variable features")
    if isinstance(n_top_features, int) and n_top_features > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key="batch", inplace=False, subset=True)
    else:
        adata = reindex(adata, n_top_features)

    logger.info("Batch specific maxabs scaling")
    # adata = batch_scale(adata, chunk_size=chunk_size)
    adata.X = MaxAbsScaler().fit_transform(adata.X)
    logger.info(f"Processed dataset shape: {adata.shape}")
    return adata


def preprocessing_atac(
    adata: ad.AnnData,
    min_features: int = 100,
    min_cells: int = 3,
    target_sum: int = 10000,
    n_top_features: int | list[str] = 100000,  # or gene list
    backed: bool = False,
) -> ad.AnnData:
    """
    Preprocessing single-cell ATAC-seq

    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 100.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization. Default: None.
    n_top_features
        Number of highly-variable features to keep. Default: 30000.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.

    Return
    -------
    The AnnData object after preprocessing.
    """
    # TODO replace this with snapatac or something from muon
    # episcanpy requires tbb, which fucks with MacOS
    # import episcanpy as epi

    logger.info("Preprocessing")
    if not isinstance(adata.X, csr.csr_matrix):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    logger.info("Filtering cells")

    sc.pp.filter_cells(adata, min_genes=min_features)

    logger.info("Filtering features")
    sc.pp.filter_genes(adata, min_cells=min_cells)

    #     adata.raw = adata
    logger.info("Finding variable features")
    if isinstance(n_top_features, int) and (n_top_features > 0) and (n_top_features < adata.shape[1]):
        pass
    else:
        adata = reindex(adata, n_top_features)

    logger.info("Batch specific maxabs scaling")
    adata.X = MaxAbsScaler().fit_transform(adata.X)
    logger.info(f"Processed dataset shape: {adata.shape}")
    return adata


def preprocessing_prot(adata: ad.AnnData, raw_adata: ad.AnnData, **kwargs) -> ad.AnnData:
    pt.pp.dsb(data=adata, data_raw=raw_adata, **kwargs)

    return adata


def batch_scale(adata: ad.AnnData) -> ad.AnnData:
    """
    Batch-specific scale data

    Parameters
    ----------
    adata
        AnnData

    Return
    ------
    AnnData
    """
    for b in adata.obs["batch"].unique():
        idx = np.where(adata.obs["batch"] == b)[0]
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        adata.X[idx] = scaler.transform(adata.X[idx])

    return adata


def reindex(adata: ad.AnnData, genes: list[str]) -> ad.AnnData:  # chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list

    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing

    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    if len(idx) == len(genes):
        adata = adata[:, genes].copy()
    else:
        new_x = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        new_x[:, idx] = adata[:, genes[idx]].X
        adata = ad.AnnData(new_x.tocsr(), obs=adata.obs, var={"var_names": genes})
    return adata


class BatchSampler(Sampler):
    """
    Batch-specific Sampler
    sampled data of each batch is from the same dataset.
    """

    def __init__(self, batch_size, batch_id, drop_last=False):
        """
        create a BatchSampler object

        Parameters
        ----------
        batch_size
            batch size for each sampling
        batch_id
            batch id of all samples
        drop_last
            drop the last samples that not up to one batch

        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_id = batch_id

    def __iter__(self):
        batch = {}
        sampler = np.random.permutation(len(self.batch_id))
        for idx in sampler:
            c = self.batch_id.iloc[idx]
            if c not in batch:
                batch[c] = []
            batch[c].append(idx)

            if len(batch[c]) == self.batch_size:
                yield batch[c]
                batch[c] = []

        for c in batch.items():
            if len(batch[c]) > 0 and not self.drop_last:
                yield batch[c]

    def __len__(self):
        if self.drop_last:
            return len(self.batch_id) // self.batch_size
        else:
            return (len(self.batch_id) + self.batch_size - 1) // self.batch_size


class SingleCellDataset(Dataset):
    """
    Dataloader of single-cell data
    """

    def __init__(self, adata, use_layer="X"):
        """
        create a SingleCellDataset object

        Parameters
        ----------
        adata
            AnnData object wrapping the single-cell data matrix
        """
        self.adata = adata
        self.shape = adata.shape
        self.use_layer = use_layer

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        if self.use_layer == "X":
            x = (
                self.adata.X[idx].squeeze().astype(float)
                if isinstance(self.adata.X[idx], np.ndarray)
                else self.adata.X[idx].toarray().squeeze().astype(float)
            )
        elif self.use_layer in self.adata.layers:
            x = self.adata.layers[self.use_layer][idx]
        else:
            x = self.adata.obsm[self.use_layer][idx]

        domain_id = self.adata.obs["batch"].cat.codes.iloc[idx]
        return x, domain_id, idx


def load_data(
    data_list: list[ad.AnnData],
    raw_data_list: list[ad.AnnData] | None = None,
    batch_categories=None,
    profile: Literal["RNA", "PROT", "ATAC"] = "RNA",
    join: str = "inner",
    batch_key: str = "batch",
    batch_name: str = "batch",
    min_features: int = 600,
    min_cells: int = 3,
    target_sum: int | None = None,
    n_top_features: int | None = None,
    backed: bool = False,
    batch_size: int = 64,
    fraction: float | None = None,
    n_obs: int | None = None,
    processed: bool = False,
    device: str = "cpu",
    num_workers: int = 4,
    use_layer: str = "X",
):
    """
    Load dataset with preprocessing

    Parameters
    ----------
    data_list
        A list of AnnData matrices to concatenate. Each matrix is referred to as a 'batch'.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_key
        Add the batch annotation to obs using this key. Default: 'batch'.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    min_features
        Filtered out cells that are detected in less than min_features. Default: 600.
    min_cells
        Filtered out genes that are detected in less than min_cells. Default: 3.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    batch_size
        Number of samples per batch to load. Default: 64.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.

    Returns
    -------
    adata
        The AnnData object after combination and preprocessing.
    trainloader
        An iterable over the given dataset for training.
    testloader
        An iterable over the given dataset for testing
    """

    if profile.upper() == "PROT" and not processed and raw_data_list:
        try:
            adata_list = [
                preprocessing(
                    adata=x,
                    raw_adata=y,
                    profile=profile,
                    min_features=min_features,
                    min_cells=min_cells,
                    target_sum=target_sum,
                    n_top_features=n_top_features,
                    backed=backed,
                )
                for x, y in zip(data_list, raw_data_list, strict=True)
            ]
        except ValueError as e:
            msg = "There is a mismatch between the filtered and raw adata lists. They should be equal in size."
            raise ValueError(msg) from e
        adata = concat_data(adata_list, batch_categories, join=join, batch_key=batch_key)
        process_batch_ident(batch_name, adata)
        adata.X = MinMaxScaler().fit_transform(adata.X)
    else:
        adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)
        if n_obs is not None or fraction is not None:
            sc.pp.subsample(adata, fraction=fraction, n_obs=n_obs)
        process_batch_ident(batch_name, adata)

        match use_layer:
            case "X" if not processed:
                adata = preprocessing(
                    adata=adata,
                    raw_adata=None,
                    profile=profile,
                    min_features=min_features,
                    min_cells=min_cells,
                    target_sum=target_sum,
                    n_top_features=n_top_features,
                    # chunk_size=chunk_size,
                    backed=backed,
                )
                adata.X = MinMaxScaler().fit_transform(
                    adata.X
                )  # Not sure why we are scaling to [-1,1] when the later VAE can only handle fitting data in range of [0,1]
            case "X" if processed:
                adata.X = MinMaxScaler().fit_transform(adata.X)
            case layer if layer in adata.layers:
                adata.layers[use_layer] = MinMaxScaler().fit_transform(adata.layers[use_layer])
            case layer if layer in adata.obsm:
                adata.obsm[use_layer] = MinMaxScaler().fit_transform(adata.obsm[use_layer])
            case _:
                msg = f"Using `{use_layer}` is not yet supported"
                raise ValueError(msg)

    scdata = SingleCellDataset(adata, use_layer=use_layer)  # Wrap AnnData into Pytorch Dataset
    trainloader = DataLoader(
        scdata,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,  # shuffle=True
        generator=Generator(device=device),
    )
    batch_sampler = BatchSampler(batch_size, adata.obs["batch"], drop_last=False)
    testloader = DataLoader(
        scdata, batch_sampler=batch_sampler, generator=Generator(device=device), num_workers=num_workers
    )

    return adata, trainloader, testloader


def process_batch_ident(batch_name: str, adata: ad.AnnData) -> None:
    if batch_name != "batch":
        if "," in batch_name:
            names = batch_name.split(",")
            adata.obs["batch"] = adata.obs[names[0]].astype(str) + "_" + adata.obs[names[1]].astype(str)
        else:
            adata.obs["batch"] = adata.obs[batch_name]
    if "batch" not in adata.obs:
        adata.obs["batch"] = "batch"
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    logger.info(f"There are {len(adata.obs['batch'].cat.categories)} batches under batch_name: {batch_name}")
