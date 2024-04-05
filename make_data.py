"""
Cleans, combines, and saves out data for stress detection.
Updated: 4/2/2024
"""
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import pandas as pd

import utils

RAW_COL_NAME = 'text'
CLEAN_COL_NAME = f'processed_{RAW_COL_NAME}'
STOPWORDS = utils.add_stopwords_missing_apostrophe()


def combine_emotion_data(folder='data', files=[
    'Stress_1.txt', 
    'Stress_2.txt', 
    'Stress_3.txt']) -> pd.DataFrame:
    dfs = []
    for file in files:
        dfs.append(
            pd.read_csv(
                os.path.join(folder, file), sep=';', header=0, names=[RAW_COL_NAME, 'emotion']
            )
        )
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame, drop_raw_col=True) -> pd.DataFrame:
    lemmatizer = WordNetLemmatizer()
    df[CLEAN_COL_NAME] = df[RAW_COL_NAME].apply(
        lambda raw_text: utils.process_text(
            text_chunk=raw_text, 
            stopwords=STOPWORDS,
            lemmatizer_obj=lemmatizer
        )
    )
    if drop_raw_col:
        df = df.drop(columns=[RAW_COL_NAME])
    return df


def make_proxy_label(raw_df: pd.DataFrame, stress_proxy: set = {
    'sadness': 1, 'anger': 1, 'fear': 1,
    'love': 0, 'joy': 0, 'surprise': 0
    }) -> pd.DataFrame:
    raw_df['label'] = raw_df['emotion'].replace(stress_proxy)
    raw_df = utils.get_slimmer_numerics(raw_df)
    return raw_df
