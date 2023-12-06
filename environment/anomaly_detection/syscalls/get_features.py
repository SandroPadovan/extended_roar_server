import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import tqdm
import pickle
import time
from typing import Any
import logging
from config import config

log_level = config.get_int("logging", "level")


def fit_vectorizer(data: list[str]) -> Any:
    """
    Creates a CountVectorizer and fits it to data with ngram_range=(1, 1).
    :return: fitted CountVectorizer
    """
    return CountVectorizer(ngram_range=(1, 1)).fit(data)


def read_raw_data(path: str = None) -> list[pd.DataFrame]:
    """
    Reads raw data into DataFrame. Path either to a single csv file or a directory of csv files.
    :param path: Path to raw data
    :return: DataFrame containing raw data
    """

    if not path:
        raise ValueError("Raw data path is None.")

    if path.endswith(".csv"):
        logging.debug('Reading raw data from single file')
        data = pd.read_csv(path)
        return [data]
    else:
        res = []
        raw_file_names = os.listdir(path)
        prog = tqdm.tqdm(total=len(raw_file_names), ncols=100, desc='Reading raw data from files',
                         disable=log_level > logging.DEBUG)
        for file_name in raw_file_names:
            if file_name.endswith(".csv"):
                data = pd.read_csv(path + file_name)
                res.append(data)
            prog.update(1)
        prog.close()
        return res


def get_features(raw_data_path: str = None, features_pkl_path: str = None,
                 store_features_pkl_path: str = None, vectorizer_path: str = None) -> pd.DataFrame:
    """
    :param raw_data_path: String path to read raw data. Must have trailing slash.
    :param features_pkl_path: String path to stored features as pickle file.
    :param store_features_pkl_path: String path to store features in pickle file.
    :param vectorizer_path: String path to stored vectorizer of training data.
    :return: DataFrame with features
    """

    # if features_pkl_path provided, read features from pickle file and return
    if features_pkl_path:
        loc = open(features_pkl_path, "rb")
        features_df = pickle.load(loc)
        return features_df

    raw_data = read_raw_data(raw_data_path)

    syscall_str = []
    for df in raw_data:
        syscall_str.append(df["syscall"].str.cat(sep=" "))

    if vectorizer_path is None:
        # build vectorizer
        vectorizer = fit_vectorizer(syscall_str)
    else:
        try:
            # read fitted vectorizer from pickle
            with open(vectorizer_path, "rb") as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)
        except (FileNotFoundError, IOError):
            # store vectorizer if it does not exist yet
            vectorizer = fit_vectorizer(syscall_str)
            with open(vectorizer_path, "wb") as vectorizer_file:
                pickle.dump(vectorizer, vectorizer_file)

    # extract frequency features using transform() of CountVectorizer
    t1 = time.time()
    frequency_features = vectorizer.transform(syscall_str)
    t = time.time() - t1
    logging.debug(f"extract features: {t:.2f} s")

    features_df = pd.DataFrame({"frequency_1gram": [np.array(row) for row in frequency_features.toarray()]})

    if store_features_pkl_path:
        features_df.to_pickle(store_features_pkl_path)
        logging.debug(f"stored features to pkl file {store_features_pkl_path}")
    return features_df


if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # --- CREATE PKL OF NORMAL FEATURES ---
    # get_features(raw_data_path='./fingerprints/evaluation/normal/syscalls/',
    #              store_features_pkl_path='./fingerprints/evaluation/normal/normal.pkl',
    #              vectorizer_path="./fingerprints/training/normal/")

    # --- CREATE PKL OF INFECTED FEATURES ---
    for rw_config in range(6):
        get_features(raw_data_path=f'./fingerprints/training/infected-c{rw_config}/syscalls/',
                     store_features_pkl_path=f'./fingerprints/training/infected-c{rw_config}/infected-c{rw_config}.pkl',
                     vectorizer_path="./fingerprints/training/normal/")
