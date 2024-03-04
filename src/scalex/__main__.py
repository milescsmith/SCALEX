#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Wed 10 Jul 2019 08:42:21 PM CST

# File Name: SCALEX.py
# Description:

"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint

from scalex import __version__
from scalex.function import SCALEX
from scalex.logging import init_logger

app = typer.Typer(
    name="scalex",
    help="Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

verbosity_level = 0


class Profile(str, Enum):
    RNA = "RNA"
    Prot = "Prot"
    ATAC = "ATAC"


class Joins(str, Enum):
    left = "left"
    right = "right"
    inner = "inner"
    outer = "outer"


def version_callback(value: bool) -> None:  # FBT001
    """Prints the version of the package."""
    if value:
        rprint(f"[yellow]scalex[/] version: [bold blue]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def verbosity(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            help="Control output verbosity. Pass this argument multiple times to increase the amount of output.",
            count=True,
        ),
    ] = 0
) -> None:
    verbosity_level = verbose  # noqa: F841


@app.callback(invoke_without_command=True)
@app.command(
    name="SCALEX",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def main(
    data_list: Annotated[
        list[Path],
        typer.Argument(
            help="A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.",
        ),
    ],
    batch_categories: Annotated[
        Optional[list[str]],
        typer.Option(
            "--batch_categories",
            "-b",
            help="Categories for the batch annotation. By default, use increasing numbers.",
        ),
    ] = None,
    profile: Annotated[
        Profile,
        typer.Option(
            "--profile",
            help="Specify the type single-cell data.",
        ),
    ] = Profile.RNA,
    batch_name: Annotated[
        str,
        typer.Option(
            "--batch_name",
            help="Use this annotation in obs as batches for training model",
        ),
    ] = "batch",
    min_features: Annotated[
        Optional[int],
        typer.Option(
            "--min_features",
            help="Filtered out cells that are detected in less than min_features.",
        ),
    ] = None,
    min_cells: Annotated[
        Optional[int],
        typer.Option(
            "--min_cells",
            help="Filtered out genes that are detected in less than min_cells.",
        ),
    ] = 3,
    join: Annotated[
        Joins,
        typer.Option(
            "--join",
            help="Use intersection ('inner') or union ('outer') of variables of different batches",
        ),
    ] = Joins.inner,
    batch_key: Annotated[
        str,
        typer.Option(
            "--batch_key",
            help="Add the batch annotation to obs using this key.",
        ),
    ] = "batch",
    n_top_features: Annotated[
        Optional[int],
        typer.Option(
            "--n_top_features",
            help="Number of highly-variable genes to keep.",
        ),
    ] = None,
    target_sum: Annotated[
        Optional[int],
        typer.Option(
            "--target_sum",
            help="After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.",
        ),
    ] = None,
    processed: Annotated[
        bool,
        typer.Option(
            "--processed",
            help="",
        ),
    ] = False,
    fraction: Annotated[
        Optional[float],
        typer.Option(
            "--fraction",
            help="",
        ),
    ] = None,
    n_obs: Annotated[
        Optional[int],
        typer.Option(
            "--n_obs",
            help="",
        ),
    ] = None,
    use_layer: Annotated[
        str,
        typer.Option(
            "--use_layer",
            help="",
        ),
    ] = "X",
    backed: Annotated[
        bool,
        typer.Option(
            "--backed",
            help="",
        ),
    ] = False,
    projection: Annotated[
        Optional[str],
        typer.Option(
            "--projection",
            "-p",
            help="",
        ),
    ] = None,
    impute: Annotated[
        bool,
        typer.Option(
            "--impute",
            help="",
        ),
    ] = False,
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            "-o",
            help="",
        ),
    ] = "output",
    learn_rate: Annotated[
        float,
        typer.Option(
            "--lr",
            help="",
        ),
    ] = 2e-4,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch_size",
            help="Number of samples per batch to load.",
        ),
    ] = 64,
    gpu: Annotated[
        int,
        typer.Option(
            "-g",
            "--gpu",
            help="Index of GPU to use if GPU is available.",
        ),
    ] = 0,
    max_iteration: Annotated[
        int,
        typer.Option(
            "--max_iteration",
            help="Max iterations for training. Training one batch_size samples is one iteration.",
        ),
    ] = 30000,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=" Random seed for torch and numpy.",
        ),
    ] = 124,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk_size",
            help="Number of samples from the same batch to transform. [bold red]NOT CURRENTLY USED[/bold red]",
        ),
    ] = 20000,
    ignore_umap: Annotated[
        bool,
        typer.Option(
            "--ignore_umap",
            help=" If True, do not perform UMAP for visualization and leiden for clustering.",
        ),
    ] = False,
    repeat: Annotated[
        bool,
        typer.Option(
            "--repeat",
            help="Use with projection. If False, concatenate the reference and projection datasets for downstream analysis. If True, only use projection datasets.",
        ),
    ] = False,
    assess: Annotated[
        bool,
        typer.Option(
            "--assess",
            help="If True, calculate the entropy_batch_mixing score and silhouette score to evaluate integration results.",
        ),
    ] = False,
    evaluate_model: Annotated[
        bool,
        typer.Option(
            "--eval",
            help="",
        ),
    ] = False,
    num_workers: Annotated[
        int,
        typer.Option(
            "--num_workers",
            help="Number of CPU cores to use.",
        ),
    ] = 4,
    show: Annotated[
        bool,
        typer.Option(
            "--show",
            help="",
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            help="Set the vebosity level: -v (only warnings) -vv (warnings and info) -vvv (warnings, info and debug)",
            count=True,
        ),
    ] = 0,
    save_log: Annotated[bool, typer.Option("-s", "--save_log", help="Save the log to a file")] = False,
    version: Annotated[  # noqa ARG007
        bool,
        typer.Option(
            "-V",
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Prints the version of the SCALEX package.",
        ),
    ] = False,  # FBT002
):
    init_logger(verbosity, save_log)

    return SCALEX(
        data_list,
        batch_categories=batch_categories,
        profile=profile,
        join=join,
        batch_key=batch_key,
        min_features=min_features,
        min_cells=min_cells,
        target_sum=target_sum,
        n_top_features=n_top_features,
        fraction=fraction,
        n_obs=n_obs,
        processed=processed,
        use_layer=use_layer,
        backed=backed,
        batch_size=batch_size,
        lr=learn_rate,
        max_iteration=max_iteration,
        impute=impute,
        batch_name=batch_name,
        seed=seed,
        gpu=gpu,
        outdir=outdir,
        projection=projection,
        chunk_size=chunk_size,
        ignore_umap=ignore_umap,
        repeat=repeat,
        verbose=verbose,
        assess=assess,
        evaluate=evaluate_model,
        num_workers=num_workers,
        show=show,
    )


if __name__ == "__main__":
    app()
