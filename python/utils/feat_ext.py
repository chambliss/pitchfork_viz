import os
import pandas as pd
import sqlite3
import spacy
from spacy.tokens.doc import Doc
import time
from typing import Tuple


create_id_column = lambda df: pd.Index(range(df.shape[0]), name="id")


def get_table_contents(table: str) -> pd.DataFrame:

    query = f"SELECT * FROM {table}"
    con = sqlite3.connect("../data/pitchfork_database.sqlite")
    df = pd.read_sql_query(query, con)
    con.close()

    if table == "content":
        df["content"] = df["content"].str.replace("\xa0", " ")

    return df


def get_unique_artists(artists_df: pd.DataFrame) -> pd.DataFrame:

    unique_artists = (
        artists_df.groupby("artist").agg(lambda x: str(x.values)).reset_index()
    )

    # Turn the string of reviewids into a list of ids
    unique_artists["reviewid"] = unique_artists["reviewid"].str.strip("[]").str.split()

    unique_artists.index = create_id_column(unique_artists)
    unique_artists.columns = ["artist", "review_ids"]

    return unique_artists


def get_unique_genres(genres_df: pd.DataFrame) -> pd.DataFrame:

    unique_genres = genres_df["genre"].drop_duplicates().to_frame("genre")

    unique_genres.index = create_id_column(unique_genres)

    return unique_genres


# I don't think I actually need this function - we can leave this info in the reviews df
def get_unique_authors(reviews_df: pd.DataFrame) -> pd.DataFrame:

    unique_authors = (
        reviews_df[["author", "author_type"]].drop_duplicates().sort_values(by="author")
    )

    unique_authors.index = create_id_column(unique_authors)

    return unique_authors


def create_ents_df(spacy_doc: Doc, review_id: int) -> pd.DataFrame:

    desired_labels = ["ORG", "PERSON", "GPE", "NORP", "EVENT"]
    ents_in_text = [
        (review_id, ent.text, ent.label_)
        for ent in spacy_doc.ents
        if ent.label_ in desired_labels
    ]

    ents_df = pd.DataFrame(ents_in_text)
    cols = ["review_id", "entity", "spacy_pred"]

    if not ents_df.empty:
        ents_df.columns = cols
        ents_df = ents_df.drop_duplicates(subset="entity")
    else:
        ents_df = pd.DataFrame(columns=cols)

    return ents_df


def create_pos_df(spacy_doc: Doc, review_id: int) -> pd.DataFrame:

    """
    Creates a DF of every adjective, noun, and verb occurring in a given
    review. Takes a spacy Doc object and returns a DataFrame.
    """

    desired_pos = ["ADJ", "NOUN", "VERB"]
    pos_tags_in_text = [
        (review_id, tok.text, tok.pos_) for tok in spacy_doc if tok.pos_ in desired_pos
    ]

    pos_df = pd.DataFrame(pos_tags_in_text).drop_duplicates()
    cols = ["review_id", "word", "pos"]

    if not pos_df.empty:
        pos_df.columns = cols
    else:
        pos_df = pd.DataFrame(columns=cols)

    return pos_df


def extract_text_features(
    content_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    start_time = time.time()

    # Prepare spacy model and document pipeline
    nlp = spacy.load("en_core_web_lg")
    review_ids, texts = content_df["reviewid"].values, content_df["content"].values
    doc_generator = nlp.pipe(texts, disable=["parser"], n_threads=16, batch_size=50)
    doc_pipeline = zip(review_ids, doc_generator)

    # Create empty dfs to append to
    combined_ents_df, combined_pos_df = pd.DataFrame(), pd.DataFrame()

    # Main loop: Extract entities and interesting words (adjs, verbs, nouns)
    # from each review. "pos": "part of speech"
    for i, (review_id, doc) in enumerate(doc_pipeline, start=1):
        combined_ents_df = combined_ents_df.append(create_ents_df(doc, review_id))
        combined_pos_df = combined_pos_df.append(create_pos_df(doc, review_id))

        # Every 1000 docs, save intermediate CSVs and report time elapsed
        if i % 1000 == 0:
            combined_ents_df.to_csv(f"../data/ents_{i}.csv")
            combined_pos_df.to_csv(f"../data/pos_{i}.csv")
            print(f"Finished {i} reviews. Time elapsed: {time.time() - start_time}s")

            # Remove the last round's temp csv files
            if os.path.exists(f"../data/ents_{i - 1000}.csv"):
                os.remove(f"../data/ents_{i - 1000}.csv")
                os.remove(f"../data/pos_{i - 1000}.csv")

    return combined_ents_df, combined_pos_df
