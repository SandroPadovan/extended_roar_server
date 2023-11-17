import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import os
import tqdm
import pickle
import time
from typing import Tuple
import numpy as np
from config import config
import logging

# FEATURES = ['frequency', 'tfidf', 'hashing', 'onehot']    # complete feature set
FEATURES = ['frequency']
log_level = config.get_int("logging", "level")


def get_syscall_dict(ngrams_dict: dict) -> dict:
    """
    Returns a dict with the count of each single term in the ngrams_dict.

    :param ngrams_dict:
    :return: dict
    """

    syscall_dict = {}
    i = 0
    for ngram in ngrams_dict:
        if len(ngram.split()) == 1:
            syscall_dict[ngram] = i
            i += 1
    return syscall_dict


def create_vectorizers(data: list[str], ngram: int):
    count_vectorizer = CountVectorizer(ngram_range=(1, ngram)).fit(data)
    logging.debug(f'create count vectorizer finished for ngram={ngram}')

    ngrams_dict = count_vectorizer.vocabulary_
    syscall_dict = get_syscall_dict(ngrams_dict)

    tfidf_vectorizer = None
    if 'tfidf' in FEATURES:
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, ngram), vocabulary=ngrams_dict, dtype=np.int8).fit(data)
        logging.debug(f'create tf-idf vectorizer finished for ngram={ngram}')

    hashing_vectorizer = None
    if 'hashing' in FEATURES:
        hashing_vectorizer = HashingVectorizer(n_features=2 ** 5, dtype=np.int8).fit(data)
        logging.debug(f'create hashing vectorizer finished for ngram={ngram}')

    return syscall_dict, ngrams_dict, count_vectorizer, tfidf_vectorizer, hashing_vectorizer


def from_trace_to_longstr(syscall_trace: list) -> str:
    tracestr = ''
    for syscall in syscall_trace:
        tracestr += str(syscall) + ' '
    return tracestr


def create_onehot_encoding(total: int, index: int) -> list:
    onehot = []
    for i in range(0, total):
        if i == index:
            onehot.append(1)
        else:
            onehot.append(0)
    return onehot


def add_unk_to_dict(syscall_dict: dict) -> Tuple[dict, dict]:
    total = len(syscall_dict)
    syscall_dict['unk'] = total
    syscall_dict_onehot = dict()
    for sc in syscall_dict:
        syscall_dict_onehot[sc] = create_onehot_encoding(total + 1, syscall_dict[sc])
    return syscall_dict, syscall_dict_onehot


def replace_with_unk(syscall_trace: list, syscall_dict: dict) -> list:
    for i, sc in enumerate(syscall_trace):
        if str(sc).lower() not in syscall_dict:
            syscall_trace[i] = 'unk'
    return syscall_trace


def trace_onehot_encoding(trace: list, syscall_dict_onehot: dict) -> list:
    encoded_trace = []
    for syscall in trace:
        syscall = str(syscall)
        syscall = syscall.lower()
        if syscall.lower() in syscall_dict_onehot:
            one_hot = syscall_dict_onehot[syscall]
        else:
            syscall = 'UNK'
            one_hot = syscall_dict_onehot[syscall]
        encoded_trace.append(one_hot)
    return encoded_trace


def get_dict_sequence(trace, term_dict):
    dict_sequence = []
    for syscall in trace:
        if syscall in term_dict:
            dict_sequence.append(term_dict[syscall])
        else:
            dict_sequence.append(term_dict['unk'])
    return dict_sequence


def build_dicts(data: list[str]) -> tuple[dict, dict]:
    """
    Builds the vectorizers and dicts using the count vectorizer, tfidf vectorizer,
    and hashing vectorizer.

    :return: vectorizers, dicts
    """
    vectorizers = {}
    dicts = {}

    for i in range(1, 6):
        syscall_dict, ngrams_dict, countvectorizer, tfidfvectorizer, hashingvectorizer \
            = create_vectorizers(data, i)

        syscall_dict, syscall_dict_onehot = add_unk_to_dict(syscall_dict)

        vectorizers['countvectorizer_ngram{}'.format(i)] = countvectorizer
        vectorizers['tfidfvectorizer_ngram{}'.format(i)] = tfidfvectorizer
        vectorizers['hashingvectorizer_ngram{}'.format(i)] = hashingvectorizer
        dicts['ngrams_dict_ngram{}'.format(i)] = ngrams_dict
        dicts['syscall_dict_ngram{}'.format(i)] = syscall_dict
        dicts['syscall_dict_onehot_ngram{}'.format(i)] = syscall_dict_onehot

    return vectorizers, dicts


def read_raw_data(path: str = None) -> list[pd.DataFrame]:
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
                         disable=log_level > 10)
        for file_name in raw_file_names:
            if file_name.endswith(".csv"):
                data = pd.read_csv(path + file_name)
                res.append(data)
            prog.update(1)
        prog.close()
        return res


def get_features(raw_data_path: str = None, features_pkl_path: str = None,
                 store_features_pkl_path: str = None, vectorizers_path: str = None) -> pd.DataFrame:
    """
    :param raw_data_path: String path to read raw data. Must have trailing slash.
    :param features_pkl_path: String path to stored features as pickle file.
    :param store_features_pkl_path: String path to store features in pickle file.
    :param vectorizers_path: String path to stored vectorizers of training data.
    :return: DataFrame with features
    """

    if features_pkl_path:
        loc = open(features_pkl_path, "rb")
        features_df = pickle.load(loc)
        return features_df

    raw_data = read_raw_data(raw_data_path)

    syscall_str = []
    for df in raw_data:
        syscall_str.append(df["syscall"].str.cat(sep=" "))

    features = []

    if vectorizers_path is None:
        # build vectorizers and dicts
        vectorizers, dicts = build_dicts(syscall_str)
    else:
        try:
            # read vectorizers and dicts from pickle
            with open(f"{vectorizers_path}vectorizers.pkl", "rb") as vectorizers_file:
                vectorizers = pickle.load(vectorizers_file)
            with open(f"{vectorizers_path}dicts.pkl", "rb") as dicts_file:
                dicts = pickle.load(dicts_file)
        except (FileNotFoundError, IOError):
            # store vectorizers and dicts
            vectorizers, dicts = build_dicts(syscall_str)
            with open(f"{vectorizers_path}vectorizers.pkl", "wb") as vectorizers_file:
                pickle.dump(vectorizers, vectorizers_file)
            with open(f"{vectorizers_path}dicts.pkl", "wb") as dicts_file:
                pickle.dump(dicts, dicts_file)

    sdName = 'syscall_dict_ngram{}'.format(1)
    shdName = 'syscall_dict_onehot_ngram{}'.format(1)
    syscall_dict = dicts[sdName]
    shd = dicts[shdName]

    dict_sequence_features = []

    if 'onehot' in FEATURES:
        one_hot_features = []
        par = tqdm.tqdm(total=len(raw_data), ncols=100, desc='Create one-hot encoding', disable=log_level > 10)
        for trace in raw_data:
            syscall_trace = replace_with_unk(trace['syscall'].to_list(), syscall_dict)
            syscall_one_hot = trace_onehot_encoding(syscall_trace, shd)
            one_hot_features.append(syscall_one_hot)
            par.update(1)
        par.close()
        features.append(one_hot_features)

    par = tqdm.tqdm(total=len(raw_data), ncols=100, desc='Get dict sequence', disable=log_level > 10)
    for trace in raw_data:
        syscall_trace = replace_with_unk(trace['syscall'].to_list(), syscall_dict)
        dict_sequence = get_dict_sequence(syscall_trace, syscall_dict)
        dict_sequence_features.append(dict_sequence)
        par.update(1)
    par.close()
    features.append(dict_sequence_features)

    def transform(vectorizer, vectorizer_name: str) -> list:
        t1 = time.time()
        features = vectorizer.transform(syscall_str)
        t = time.time() - t1
        logging.debug(f"transform {vectorizer_name}: {t:.2f} s")
        return features.toarray()

    for i in range(1, 6):
        if 'frequency' in FEATURES:
            count_vectorizer_name = 'countvectorizer_ngram{}'.format(i)
            count_vectorizer = vectorizers[count_vectorizer_name]
            frequency_features = transform(count_vectorizer, count_vectorizer_name)
            features.append(frequency_features)

        if 'tfidf' in FEATURES:
            tfidf_vectorizer_name = 'tfidfvectorizer_ngram{}'.format(i)
            tfidf_vectorizer = vectorizers[tfidf_vectorizer_name]
            tfidf_features = transform(tfidf_vectorizer, tfidf_vectorizer_name)
            features.append(tfidf_features)

        if 'hashing' in FEATURES:
            hashing_vectorizer_name = 'hashingvectorizer_ngram{}'.format(i)
            hashing_vectorizer = vectorizers[hashing_vectorizer_name]
            hashing_features = transform(hashing_vectorizer, hashing_vectorizer_name)
            features.append(hashing_features)

    features_df = pd.DataFrame(features).transpose()

    # column names for complete feature set
    # features_df.columns = ['one hot encoding', 'dict index encoding',
    #                        'system calls frequency_1gram', 'system calls tfidf_1gram', 'system calls hashing_1gram',
    #                        'system calls frequency_2gram', 'system calls tfidf_2gram', 'system calls hashing_2gram',
    #                        'system calls frequency_3gram', 'system calls tfidf_3gram', 'system calls hashing_3gram',
    #                        'system calls frequency_4gram', 'system calls tfidf_4gram', 'system calls hashing_4gram',
    #                        'system calls frequency_5gram', 'system calls tfidf_5gram', 'system calls hashing_5gram'
    #                        ]
    features_df.columns = ['dict index encoding',
                           'system calls frequency_1gram',
                           'system calls frequency_2gram',
                           'system calls frequency_3gram',
                           'system calls frequency_4gram',
                           'system calls frequency_5gram',
                           ]

    if store_features_pkl_path:
        features_df.to_pickle(store_features_pkl_path)
        logging.debug('stored features to pkl file')
    return features_df
