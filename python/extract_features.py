from utils.feat_ext import (
    get_table_contents,
    create_id_column,
    get_unique_artists,
    get_unique_genres,
    create_ents_df,
    create_pos_df,
    extract_text_features,
)
import os
import pandas as pd
import sqlite3
import spacy
from spacy.tokens.doc import Doc  # mypy/typing import
import time

if __name__ == "__main__":

    # Fetch contents of each table from the database
    tables = ["artists", "content", "genres"]
    table_dfs = {table: get_table_contents(table) for table in tables}

    # Write out DFs for unique artists and unique genres
    get_unique_artists(table_dfs["artists"]).to_csv(f"../data/artists.csv")
    get_unique_genres(table_dfs["genres"]).to_csv(f"../data/genres.csv")

    # Extract text-based features for visualization later
    combined_ents_df, combined_pos_df = extract_text_features(table_dfs["content"])

    # Write out the final results
    combined_ents_df.reset_index(drop=True).to_csv("../data/review_entities.csv")
    combined_pos_df.reset_index(drop=True).to_csv("../data/review_poses.csv")

    # Remove the last couple of tempfiles
    filenames = [fn for fn in os.listdir("../data") if "ents_" in fn or "pos_" in fn]
    for tempfile in filenames:
        os.remove(f"../data/{tempfile}")
