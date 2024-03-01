from multiprocessing import freeze_support

import scanpy as sc
from scalex.function import SCALEX

adata = sc.read_h5ad("/Users/milessmith/workspace/SCALEX/test/data/merged_cite.h5ad")
adata = SCALEX(
    data_list=adata,
    batch_name="source",
    outdir="output_cpu",
    min_features=0,
    min_cells=0
    )
