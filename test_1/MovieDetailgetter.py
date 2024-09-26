#!/usr/bin/env python3
"""
MovieDetailgetter.py

Usage:
    MovieDetailgetter.py db <DB_NAME> ( --dump | --create )
    MovieDetailgetter.py query <DB_NAME> --movie_title=<MOVIE_TITLE> [--get_field=<FIELD_NAME>] [--formatter=<FORMATTER>]

Options:
    -h, --help                       Show this screen.
    -d, --dump                       Print the database content.
    -c, --create                     Create a new database.
    -m, --movie_title=<MOVIE_TITLE>   Search for a specific movie by its title.
    -g, --get_field=<FIELD_NAME>      Specify the field to retrieve [default: rating].
    -f, --formatter=<FORMATTER>       Specify an output formatter for the query [default: default].

Description:
  MovieDetailgetter.py allows interaction with an IMDb database (Top 250 movies). 
  The 'db' command supports dumping or creating a database.
  The 'query' command searches for a specific movie by its title, with optional customizations 
  to retrieve specific fields and format the output.

  Formatters:
  - "default":
    Prints a formatted string with the specified field and value:

    $ MovieDetailgetter.py query imdb_top_250_movies.csv --movie_title 'The Shawshank Redemption'
    The Shawshank Redemption: rating is 9.3

  - "funny":
    Renders the movie rating using the cowsay library:

    $ MovieDetailgetter.py query imdb_top_250_movies.csv --movie_title 'The Shawshank Redemption' --get_field year --formatter funny
    <cowsay string>
"""
from collections.abc import Callable, Generator
from functools import reduce, wraps
from itertools import chain, starmap

import imdb
import pandas as pd
from cowsay import cowsay
from docopt import docopt

FORMATTER_REGISTER = {}
SUBCOMMAND_REGISTER = {}


def add_to_register(register: dict[str, Callable], name: str) -> Callable:
    """A helper decorator to update registers."""

    def decorator(_function: Callable):
        register.update({name: _function})

        @wraps(_function)
        def wrapper(*args, **kwargs):
            return _function(*args, **kwargs)

        return wrapper

    return decorator


def formatter(name: str) -> Callable[[str], Callable]:
    """Decorator for tracking formatters."""
    return add_to_register(register=FORMATTER_REGISTER, name=name)


def subcommand(name: str) -> Callable[[str], Callable]:
    """Decorator for tracking subcommands."""
    return add_to_register(register=SUBCOMMAND_REGISTER, name=name)


class NoMatchFoundError(Exception):
    """Custom exception to set return status 1 in case no movie title is found."""

    pass


def get_top250_movies() -> list[imdb.Movie.Movie]:
    """Retrieve the top 250 movies."""
    return imdb.Cinemagoer().get_top250_movies()


def get_movie_using_id(imdb_id: str) -> imdb.Movie.Movie:
    """Retrieve a movie using its IMDb ID."""
    return imdb.Cinemagoer().get_movie(imdb_id)


def take_n_actors_from_cast(
    cast: list[imdb.Person.Person], take_max_actors: int = 3
) -> list[str]:
    """Retrieve a maximum of n actors from the cast to build the stars list."""
    return cast[0:take_max_actors]


def extract_top_actors_from_movie(movie: imdb.Movie.Movie) -> str:
    """
    Extract a list of top actors from a movie.

    IMDb movie objects can have empty 'stars' and 'cast' fields,
    so we create fallback lists and return an empty list if no actors are found.
    """
    sources = [
        lambda: movie.get("stars"),
        lambda: list(
            map(lambda person: person["name"], take_n_actors_from_cast(movie["cast"]))
        ),
    ]
    return ", ".join(
        next(filter(bool, map(lambda get_list: get_list(), sources)), [""])
    )


def extract_info_from_movie_as_dict(
    movie: imdb.Movie.Movie,
) -> dict[str, str | list[str] | None]:
    """Extract movie information as a dictionary."""
    dataframe_header_vs_imdb_mapper = {
        "place": "top 250 rank",
        "movie_title": "title",
        "rating": "rating",
        "year": "year",
        "star_cast": "stars",
    }
    return {
        **{k: movie.get(v) for k, v in dataframe_header_vs_imdb_mapper.items()},
        # TODO (hugo.avila): Cast data is no longer included in the IMDb Top 250 list (2024/09/23),
        # so we now retrieve cast data for each movie separately. This impacts performance.
        # Consider removing it if it is no longer necessary.
        "star_cast": extract_top_actors_from_movie(get_movie_using_id(movie.getID())),
    }


def top250_movies_as_df() -> pd.DataFrame:
    """Parse IMDb data and convert it into a DataFrame."""
    return pd.DataFrame(
        map(extract_info_from_movie_as_dict, get_top250_movies())
    ).fillna("")


def sanitize_arguments(args: dict[str, str]) -> dict[str, str]:
    """Standardize the arguments for use as Python variable names."""

    def clean_string(string: str) -> str:
        """Remove special characters from arguments and convert to lowercase."""
        return reduce(lambda s, r: s.replace(r, ""), "-<>", string).lower()

    return {clean_string(k): v for k, v in args.items()}


def iter_file(file_name: str) -> Generator[str, None, None]:
    """Helper function to iterate through a file line by line."""
    with open(file_name) as f:
        yield from map(lambda line: line.rstrip("\n"), f)


@subcommand(name="db")
def database_handler(create: bool, dump: bool, *args, **kwargs) -> None:
    """Subcommand to handle database operations."""

    if create:
        top250_movies_as_df().to_csv(IMDB_CSV, index=False)
        return

    if dump:
        print(*iter_file(IMDB_CSV), sep="\n")
        return

    raise RuntimeError("No subcommand was executed!")


@formatter(name="default")
def default_formatter(
    movie_title: str, **cols: dict[str, str]
) -> Generator[str, None, None]:
    """Default formatter."""
    yield from starmap(lambda k, v: f"{movie_title}: {k} is {v}!", cols.items())


@formatter(name="funny")
def funny(movie_title: str, **cols: dict[str, str]) -> Generator[str, None, None]:
    """Formatter that uses the cowsay library for fun output."""
    yield from map(cowsay, default_formatter(movie_title=movie_title, **cols))


def load_df(db_name: str) -> pd.DataFrame:
    """Helper function to load a DataFrame and handle potential errors."""
    df = pd.read_csv(db_name, dtype="str")
    assert not df.empty, "Database is empty!"
    return df


@subcommand(name="query")
def query_database(
    movie_title: str, get_field: list[str], formatter: str, *args, **kwargs
) -> None:
    """Subcommand to handle query operations."""
    df = (
        load_df(IMDB_CSV)
        # Filter by movie_title
        .loc[lambda df_: df_.movie_title.eq(movie_title)]
        # Slice the DataFrame (discard duplicates but keep order)
        .loc[:, list({k: 0 for k in ["movie_title", get_field]}.keys())]
    )

    if df.empty:
        raise NoMatchFoundError(f'No matches found for the title "{movie_title}"!')

    print(
        *chain.from_iterable(df.apply(lambda cols: USER_FORMATTER(**cols), axis=1)),
        sep="\n",
    )


if __name__ == "__main__":
    args = sanitize_arguments(docopt(__doc__))
    IMDB_CSV = args["db_name"]

    match args:
        case {"db": True}:
            SUBCOMMAND_REGISTER["db"](**args)

        case {"query": True}:
            assert (
                args["formatter"] in FORMATTER_REGISTER
            ), f'Formatter {USER_FORMATTER} is unknown. Expected: {", ".join(FORMATTER_REGISTER.keys())}'
            assert (
                args["get_field"] != "movie_title"
            ), 'The value "get_field" cannot be "movie_title".'

            USER_FORMATTER = FORMATTER_REGISTER[args["formatter"]]
            SUBCOMMAND_REGISTER["query"](**args)

        case _:
            raise RuntimeError(
                f'None of the registered subcommands were found! Expected: {", ".join(SUBCOMMAND_REGISTER.keys())}'
            )
