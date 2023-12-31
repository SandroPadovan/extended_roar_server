import time
import logging
from sklearn.model_selection import train_test_split
from sklearn import metrics
from environment.anomaly_detection.syscalls.get_features import get_features
from environment.anomaly_detection.constructor import get_syscall_classifier
from config import config
import numpy as np


def fit_to_data(train_data: list, test_data: list) -> tuple[float, float, float, float]:
    classifier = get_syscall_classifier()

    y = [1 for _ in range(0, len(train_data))]

    shuffle = config.get_default_bool("anomaly_detection", "shuffle_training_data", default=False)
    X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=.3, shuffle=shuffle, random_state=42)
    y_val = [1 for _ in range(0, len(X_val))]

    t1 = time.time()
    classifier.fit(X_train)
    train_t = time.time() - t1

    y_pred = classifier.predict(X_val)
    val_score = metrics.accuracy_score(y_val, y_pred)

    y_test = [-1 for _ in range(0, len(test_data))]

    t1 = time.time()
    y_pred = classifier.predict(test_data)
    t_test = time.time() - t1

    test_score = metrics.accuracy_score(y_test, y_pred)
    return val_score, test_score, train_t, t_test


def train_syscall_anomaly_detection() -> tuple[float, float, float, float]:
    """
    Trains the syscall classifier using the training data at the configured path.
    Tests the classifier using testing data and returns the accuracy score of the validation and test set,
    as well as the training time and the test prediction time.

    :return: validation accuracy score, test accuracy score, training time, test prediction time
    """
    logging.debug('training syscall AD...')

    training_data_path = config.get('anomaly_detection', 'syscall_training_path')
    test_data_path = config.get('anomaly_detection', 'syscall_test_path')
    vectorizer_path = config.get('anomaly_detection', 'normal_vectorizer_path')

    if training_data_path.endswith('.pkl') and test_data_path.endswith('.pkl'):
        normal_data_df = get_features(features_pkl_path=training_data_path)
        infected_data_df = get_features(features_pkl_path=test_data_path)
    else:
        normal_data_df = get_features(raw_data_path=training_data_path, vectorizer_path=vectorizer_path)
        infected_data_df = get_features(raw_data_path=test_data_path, vectorizer_path=vectorizer_path)

    feature_name = config.get('anomaly_detection', 'syscall_feature')

    normal_data_df = normal_data_df[feature_name].tolist()
    infected_data_df = infected_data_df[feature_name].tolist()

    return fit_to_data(normal_data_df, infected_data_df)


def detect_syscall_anomaly(syscall_feature: list) -> list[int]:
    """
    Classifies the syscall features into normal and anomalous, returns the predictions.

    :param syscall_feature: extracted syscall features
    :return: list of predictions: 1=normal, -1=anomalous
    """
    clf = get_syscall_classifier()
    pred = clf.predict(syscall_feature)
    return pred.tolist()


if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s: %(message)s', level=config.get_int("logging", "level"))
    feature_name = config.get('anomaly_detection', 'syscall_feature')

    behaviors = ["infected", "compression", "installation"]
    pkl_file_identifier = "c+n"

    # number of evaluation runs
    n = 100

    for behavior in behaviors:
        val_scores = []
        test_scores = []
        normal_count = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
        }
        anomalous_count = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
        }
        percentage = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
        }

        for i in range(n):

            # train AD
            val_score, test_score, train_t, t_test = train_syscall_anomaly_detection()
            val_scores.append(val_score)
            test_scores.append(test_score)

            for j in range(6):
                features = get_features(
                    features_pkl_path=f'./fingerprints/training/{behavior}-c{j}/{pkl_file_identifier}-c{j}.pkl')

                data = features[feature_name].tolist()

                # prediction
                pred = detect_syscall_anomaly(data)

                normal_count[j].append(pred.count(1))
                anomalous_count[j].append(pred.count(-1))
                percentage[j].append(pred.count(1)/len(pred) * 100)


        print(f"val_score: {np.mean(val_scores):.2f} %")
        print(f"test_score: {np.mean(test_scores):.2f} %")

        for i in range(6):
            print(f'syscall predictions c{i}: {np.mean(normal_count[i])} normal, {np.mean(anomalous_count[i])} anomalous \t '
                  f'-> {np.mean(percentage[i]):.2f} % normal')
