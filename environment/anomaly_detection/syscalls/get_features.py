import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import os
import tqdm
import pickle
import time
from typing import Tuple


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


def create_vectorizers(corpus: list[str], ngram: int):
    count_vectorizer = CountVectorizer(ngram_range=(1, ngram)).fit(corpus)
    print(f'create count vectorizer finished for ngram={ngram}')

    ngrams_dict = count_vectorizer.vocabulary_
    syscall_dict = get_syscall_dict(ngrams_dict)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, ngram), vocabulary=ngrams_dict).fit(corpus)
    print(f'create tf-idf vectorizer finished for ngram={ngram}')

    hashing_vectorizer = HashingVectorizer(n_features=2 ** 5).fit(corpus)
    print(f'create hashing vectorizer finished for ngram={ngram}')

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


def build_dicts(corpus: list[str]) -> tuple[dict, dict]:
    """
    Builds the vectorizers and dicts using the count vectorizer, tfidf vectorizer,
    and hashing vectorizer.

    :return: vectorizers, dicts
    """
    vectorizers = {}
    dicts = {}

    for i in range(1, 6):
        syscall_dict, ngrams_dict, countvectorizer, tfidfvectorizer, hashingvectorizer \
            = create_vectorizers(corpus, i)

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
        print('Reading raw data from single file')
        data = pd.read_csv(path)
        return [data]
    else:
        res = []
        raw_file_names = os.listdir(path)
        prog = tqdm.tqdm(total=len(raw_file_names), ncols=100, desc='Reading raw data from files')
        for file_name in raw_file_names:
            if file_name.endswith(".csv"):
                data = pd.read_csv(path + file_name)
                res.append(data)
            prog.update(1)
        prog.close()
        return res


def get_features(raw_data_path: str = None, features_pkl_path: str = None,
                 store_features_pkl_path: str = None) -> pd.DataFrame:
    """

    :param raw_data_path:
    :param features_pkl_path:
    :param store_features_pkl_path:
    :return:
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

    vectorizers, dicts = build_dicts(syscall_str)

    ngram_dict_name = 'ngrams_dict_ngram{}'.format(1)
    sdName = 'syscall_dict_ngram{}'.format(1)
    shdName = 'syscall_dict_onehot_ngram{}'.format(1)
    ngram_dict = dicts[ngram_dict_name]
    syscall_dict = dicts[sdName]
    shd = dicts[shdName]

    one_hot_features = []
    dict_sequence_features = []

    t1 = time.time()
    par = tqdm.tqdm(total=len(raw_data), ncols=100, desc='Create one-hot encoding')
    for trace in raw_data:
        syscall_trace = replace_with_unk(trace['syscall'].to_list(), syscall_dict)
        syscall_one_hot = trace_onehot_encoding(syscall_trace, shd)
        one_hot_features.append(syscall_one_hot)
        par.update(1)
    par.close()
    key = 'syscall_one_hot'
    t = time.time() - t1
    print(f"{key}: {t:.2f} s")
    # pca_name = key + '_pca'
    # inputs = padding_onehot(one_hot_features, 160000)
    # one_hot_features_pca, pca = get_pca_feature(inputs)
    # pcas[pca_name] = pca
    features.append(one_hot_features)
    # features.append(one_hot_features_pca)
    par = tqdm.tqdm(total=len(raw_data), ncols=100, desc='Get dict sequence')
    t1 = time.time()
    for trace in raw_data:
        syscall_trace = replace_with_unk(trace['syscall'].to_list(), syscall_dict)
        dict_sequence = get_dict_sequence(syscall_trace, syscall_dict)
        dict_sequence_features.append(dict_sequence)
        par.update(1)
    par.close()
    t = time.time() - t1
    key = 'dict_sequence'
    print(f"{key}: {t:.2f} s")
    # pca_name = key + '_pca'
    # inputs = padding_dictencoding(dict_sequence_features, 160000)
    # dict_sequence_pca, pca = get_pca_feature(inputs)
    # pcas[pca_name] = pca
    features.append(dict_sequence_features)

    # features.append(dict_sequence_pca)

    def transform(vectorizer, vectorizer_name: str) -> list:
        t1 = time.time()
        features = vectorizer.transform(syscall_str)
        t = time.time() - t1
        print(f"{vectorizer_name}: {t:.2f} s")
        return features.toarray()

    for i in range(1, 6):
        count_vectorizer_name = 'countvectorizer_ngram{}'.format(i)
        count_vectorizer = vectorizers[count_vectorizer_name]
        frequency_features = transform(count_vectorizer, count_vectorizer_name)
        features.append(frequency_features)
        # frequency_pca_name = key + '_pca'
        # frequency_pca,pca = get_pca_feature(frequency_features)
        # pcas[frequency_pca_name] = pca
        # features.append(frequency_pca)

        tfidf_vectorizer_name = 'tfidfvectorizer_ngram{}'.format(i)
        tfidf_vectorizer = vectorizers[tfidf_vectorizer_name]
        tfidf_features = transform(tfidf_vectorizer, tfidf_vectorizer_name)
        features.append(tfidf_features)
        # tfidf_pca_name = key + '_pca'
        # tfidf_pca,pca = get_pca_feature(tfidf_features)
        # pcas[tfidf_pca_name] = pca
        # features.append(tfidf_pca)

        hashing_vectorizer_name = 'hashingvectorizer_ngram{}'.format(i)
        hashing_vectorizer = vectorizers[hashing_vectorizer_name]
        hashing_features = transform(hashing_vectorizer, hashing_vectorizer_name)
        features.append(hashing_features)

    features_df = pd.DataFrame(features).transpose()
    features_df.columns = ['one hot encoding', 'dict index encoding',
                           'system calls frequency_1gram', 'system calls tfidf_1gram', 'system calls hashing_1gram',
                           'system calls frequency_2gram', 'system calls tfidf_2gram', 'system calls hashing_2gram',
                           'system calls frequency_3gram', 'system calls tfidf_3gram', 'system calls hashing_3gram',
                           'system calls frequency_4gram', 'system calls tfidf_4gram', 'system calls hashing_4gram',
                           'system calls frequency_5gram', 'system calls tfidf_5gram', 'system calls hashing_5gram'
                           ]

    if store_features_pkl_path:
        features_df.to_pickle(store_features_pkl_path)
    return features_df
