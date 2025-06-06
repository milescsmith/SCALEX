from multiprocessing import freeze_support

import scanpy as sc

from scalex.function import SCALEX

# adata = sc.read("/Users/milessmith/workspace/SCALEX/test/data/merged_cite.h5ad")
scatac_1_cite = sc.read("/home/milo/workspace/SCALEX/test/data/scatac_1_prot.h5ad")
scatac_2_cite = sc.read("/home/milo/workspace/SCALEX/test/data/scatac_2_prot.h5ad")

adata = SCALEX(
    data_list=[scatac_1_cite, scatac_2_cite],
    batch_name="sample",
    outdir="output_cpu",
    profile="prot",
    processed=True,
    use_layer="X",
    min_features=0,
    min_cells=0
    )
