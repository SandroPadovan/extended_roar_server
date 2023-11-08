import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from environment.anomaly_detection.syscalls.get_features import get_features
from environment.anomaly_detection.constructor import get_syscall_classifier
from config import config


def fit_to_data(train_data: list, test_data: list) -> tuple[float, float, float, float]:
    classifier = get_syscall_classifier()

    y = [1 for _ in range(0, len(train_data))]
    X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=.3, shuffle=False)
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

    training_data_path = config.get('anomaly_detection', 'syscall_training_path')
    test_data_path = config.get('anomaly_detection', 'syscall_test_path')

    if training_data_path.endswith('.pkl') and test_data_path.endswith('.pkl'):
        normal_data_df = get_features(features_pkl_path=training_data_path)
        infected_data_df = get_features(features_pkl_path=test_data_path)
    else:
        normal_data_df = get_features(raw_data_path=training_data_path)
        infected_data_df = get_features(raw_data_path=test_data_path)

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
    return pred
