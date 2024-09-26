#!/usr/bin/env python3
"""
Part 2 solver

Usage:
    part2_solver.py <COORDS_TSV> <QUERIES_TSV>

Arguments:
    <COORDS_TSV>   Path to the TSV file containing coordinates.
    <QUERIES_TSV>  Path to the TSV file containing queries.

Options:
    -h --help  Show this screen.
"""
import re
import sys
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import lru_cache, partial, reduce
from itertools import starmap
from operator import getitem

import pandas as pd  # type: ignore
from docopt import docopt  # type: ignore
from toolz import curry  # type: ignore
from toolz.functoolz import compose  # type: ignore

curr_map = curry(map)
curr_starmap = curry(starmap)
curr_filter = curry(filter)


@dataclass(frozen=True, slots=True)
class Expr:
    """Type to store the CIGAR mini language."""

    procedure: str
    step: int


def parse_coords_file(coords_file: str) -> pd.DataFrame:
    """Helper function to load the input coords file."""
    return pd.read_csv(
        coords_file,
        sep="\t",
        header=None,
        names=["transcript", "chromosome", "position", "cigar"],
    ).astype(dict(position=int))


def parse_query_file(query_file: str) -> pd.DataFrame:
    """Helper function to load the input query file."""
    return pd.read_csv(
        query_file,
        sep="\t",
        header=None,
        names=["transcript", "position"],
    ).astype(dict(position=int))


def cigar_parser(cigar: str) -> Generator[Expr, None, None]:
    """Function to convert a CIGAR string into a generator of expressions."""
    # TODO (Hugo Avila): re.findall has an eager impl, switch to some lazy method
    # if you have some time later.
    def findall(regex: str) -> Callable[[str], list[str]]:
        """Stub findall all to easy composition."""
        return partial(re.findall, regex)

    def extract_tokens(cigar: str) -> list[str]:
        """Preload token regex."""
        return findall(r"\d+\w")(cigar)

    def tokens_to_expression(instruction: str) -> Expr:
        """Preload procedure and step regex."""
        return Expr(
            procedure=findall(r"\w$")(instruction)[0],
            step=int(findall(r"^\d+")(instruction)[0]),
        )

    yield from compose(
        curr_map(tokens_to_expression),
        extract_tokens,
    )(cigar)


# Some times we have more than one query for the same transcript
# let's create a small cache to store the "positions" list.
@lru_cache(maxsize=128)
def generate_alignment(
    cigar_string: str, target_pos, query_pos
) -> list[tuple[int | None, int | None]]:
    """This function will do the generation of the alingments. It basically
    works like a very simple eval interpreter for the CIGAR mini language.
    """
    positions = [(target_pos, query_pos)]

    def get_max_from_tuple(idx: int) -> int:
        return max(
            filter(lambda x: x is not None, map(lambda x: getitem(x, idx), positions))
        )

    for expr in cigar_parser(cigar_string):
        match expr:
            case Expr(procedure="M"):
                strategy = lambda last_target_pos, last_query_pos: (
                    last_target_pos + 1,
                    last_query_pos + 1,
                )
            case Expr(procedure="D"):
                strategy = lambda last_target_pos, last_query_pos: (
                    last_target_pos + 1,
                    None,
                )
            case Expr(procedure="I"):
                strategy = lambda last_target_pos, last_query_pos: (
                    None,
                    last_query_pos + 1,
                )
            case _:
                raise RuntimeError(f"Unknown cigar procedure: {expr.procedure}!")

        for _ in range(expr.step):
            last_target_pos = get_max_from_tuple(0)
            last_query_pos = get_max_from_tuple(1)
            positions.append(strategy(last_target_pos, last_query_pos))

    return positions


def get_matching_pos(
    positions: list[tuple[int | None, int | None]], query_pos
) -> tuple[int, int]:
    """Function to filter the position list and pick a query mapping."""
    return compose(
        next,
        curr_filter(lambda _tuple: _tuple[1] == query_pos),
        curr_filter(
            lambda _tuple: all(map(lambda val: val is not None, _tuple)),
        ),
    )(positions)


def sanitize_arguments(args: dict[str, str]) -> dict[str, str]:
    """Standardize the arguments for use as Python variable names."""

    def clean_string(string: str) -> str:
        """Remove special characters from arguments and convert to lowercase."""
        return reduce(lambda s, r: s.replace(r, ""), "-<>", string).lower()

    return {clean_string(k): v for k, v in args.items()}


def main(coords_tsv: str, queries_tsv: str) -> None:
    """Creates the output tsv."""
    (
        # Parse coords file
        parse_coords_file(coords_tsv)
        # Inner join coords with the queries.
        .merge(
            # Parse query files.
            parse_query_file(queries_tsv),
            on=["transcript"],
            how="inner",
            suffixes=("_targets", "_queries"),
        )
        # Run the queries.
        .assign(
            map_chrm_pos=lambda df_: (
                df_.loc[:, ["cigar", "position_targets", "position_queries"]].apply(
                    lambda cols: get_matching_pos(
                        positions=partial(generate_alignment, query_pos=0)(
                            cigar_string=cols.cigar, target_pos=cols.position_targets
                        ),
                        query_pos=cols.position_queries,
                    ),
                    axis="columns",
                )
            )
        )
        # Split the results in two columns.
        .assign(
            final_target_pos=lambda df_: df_.map_chrm_pos.apply(
                lambda _tuple: _tuple[0]
            ),
            final_query_pos=lambda df_: df_.map_chrm_pos.apply(
                lambda _tuple: _tuple[1]
            ),
        )
        # Create the output.
        .loc[
            :,
            [
                "transcript",
                "final_query_pos",
                "chromosome",
                "final_target_pos",
            ],
        ]
        # Send the table to stdout
        .to_csv(sys.stdout, index=False, header=None, sep="\t")
    )


if __name__ == "__main__":
    main(**sanitize_arguments(docopt(__doc__)))
