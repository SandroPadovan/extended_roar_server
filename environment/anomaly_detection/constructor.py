from pyod.models.iforest import IForest
import logging

from environment.anomaly_detection.advanced_preprocessor import AdvancedPreprocessor
from environment.anomaly_detection.autoencoder import AutoEncoder
from environment.anomaly_detection.simple_preprocessor import SimplePreprocessor
from environment.settings import MAX_ALLOWED_CORRELATION_AE, MAX_ALLOWED_CORRELATION_IF
from environment.state_handling import get_prototype
from config import config
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

RESOURCE_CONTAMINATION_FACTOR = config.get_float('anomaly_detection', 'resource_contamination_factor')
SYSCALL_CONTAMINATION_FACTOR = config.get_float('anomaly_detection', 'syscall_contamination_factor')
SYSCALL_CLASSIFIER_ALGO = config.get("anomaly_detection", "syscall_classifier_algo")

# ========================================
# ==========   GLOBALS   ==========
# ========================================
CLASSIFIER = None
SYSCALL_CLASSIFIER = None
PREPROCESSOR = None


def get_preprocessor():
    global PREPROCESSOR
    if not PREPROCESSOR:
        proto = get_prototype()
        if proto in ["1", "2", "99"]:
            PREPROCESSOR = SimplePreprocessor()
        elif proto in ["3", "4", "5", "6", "7", "8", "9", "10", "98"]:
            PREPROCESSOR = AdvancedPreprocessor(__get_correlation_threshold())
        else:
            logging.warning("Unknown prototype. Falling back to default simple preprocessor!")
            PREPROCESSOR = SimplePreprocessor()
    return PREPROCESSOR


def reset_AD():
    global PREPROCESSOR
    PREPROCESSOR = None
    global CLASSIFIER
    CLASSIFIER = None
    global SYSCALL_CLASSIFIER
    SYSCALL_CLASSIFIER = None


def get_classifier():
    global CLASSIFIER
    if not CLASSIFIER:
        proto = get_prototype()
        if proto in ["1", "2", "3", "4", "5", "6", "7", "8", "10", "98", "99"]:
            CLASSIFIER = IForest(random_state=42, contamination=RESOURCE_CONTAMINATION_FACTOR)
        elif proto in ["9"]:
            CLASSIFIER = AutoEncoder(encoding_dim=[40, 20, 10, 20, 40], random_state=42,
                                     outlier_percentage=RESOURCE_CONTAMINATION_FACTOR)
        else:
            logging.warning("Unknown prototype. Falling back to Isolation Forest classifier!")
            CLASSIFIER = IForest(random_state=42, contamination=RESOURCE_CONTAMINATION_FACTOR)
    return CLASSIFIER


def get_syscall_classifier():
    global SYSCALL_CLASSIFIER
    if SYSCALL_CLASSIFIER is None:

        if SYSCALL_CLASSIFIER_ALGO == "LOF":
            # additional benign behaviors used
            SYSCALL_CLASSIFIER = LocalOutlierFactor(n_neighbors=3400, contamination=SYSCALL_CONTAMINATION_FACTOR,
                                                    novelty=True)
        else:
            # no additional behavior used
            SYSCALL_CLASSIFIER = IsolationForest(contamination=SYSCALL_CONTAMINATION_FACTOR, random_state=42)

    return SYSCALL_CLASSIFIER


def __get_correlation_threshold():
    proto = get_prototype()
    if proto in ["9"]:
        return MAX_ALLOWED_CORRELATION_AE
    else:
        return MAX_ALLOWED_CORRELATION_IF
