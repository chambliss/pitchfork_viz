import os
import pandas as pd
import sqlite3
import spacy
from spacy.tokens.doc import Doc
import time


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


def extract_text_features(content_df: pd.DataFrame, pos_file: str, ents_file: str):

    """
    Extracts the parts-of-speech and entity features using spaCy. This function
    does not return anything, it just writes the files out to the specified paths
    in the /data/ directory.
    """

    start_time = time.time()

    # Prepare spaCy model and document pipeline
    nlp = spacy.load("en_core_web_lg")
    review_ids, texts = content_df["reviewid"].values, content_df["content"].values
    doc_generator = nlp.pipe(texts, disable=["parser"], batch_size=32)
    doc_pipeline = zip(review_ids, doc_generator)
    pos_path, ents_path = f"../data/{pos_file}", f"../data/{ents_file}"

    with open(ents_path, "w") as pos_file, open(pos_path, "w") as ents_file:

        for i, (review_id, doc) in enumerate(doc_pipeline, start=1):

            desired_pos = ["ADJ", "NOUN", "VERB"]
            pos_tags_in_text = [
                ",".join([str(review_id), tok.text, tok.pos_, "\n"])
                for tok in doc
                if tok.pos_ in desired_pos
            ]

            desired_labels = ["ORG", "PERSON", "GPE", "NORP", "EVENT"]
            ents_in_text = [
                ",".join([str(review_id), ent.text, ent.label_, "\n"])
                for ent in doc.ents
                if ent.label_ in desired_labels
            ]

            pos_file.writelines(pos_tags_in_text)
            ents_file.writelines(ents_in_text)

            # Every 1000 docs, report time elapsed
            if i % 1000 == 0:
                print(
                    f"Finished {i} reviews. Time elapsed: {time.time() - start_time}s"
                )

    return None
