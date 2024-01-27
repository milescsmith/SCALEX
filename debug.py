from scalex.function import SCALEX

adata = SCALEX(
    data_list="merged_cite.h5ad",
    batch_name="source",
    outdir="output/",
    min_features=0,
    min_cells=0
    )
