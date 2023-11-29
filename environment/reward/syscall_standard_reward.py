import logging
from environment.reward.abstract_reward import AbstractReward
from environment.anomaly_detection.syscall_anomaly_detection import train_syscall_anomaly_detection, \
    detect_syscall_anomaly


class SyscallStandardReward(AbstractReward):
    def __init__(self, r_done, r_hidden, r_detected):
        self.r_done = r_done
        self.r_hidden = r_hidden
        self.r_detected = r_detected

    @staticmethod
    def prepare_reward_computation():
        train_syscall_anomaly_detection()

    def compute_reward(self, syscall_features, done):
        if done:
            return self.r_done

        prediction = detect_syscall_anomaly(syscall_features)[0]  # int [-1, 1]
        logging.info("--- Detected {} Syscall.".format("anomalous" if prediction == -1 else "normal"))
        if bool(prediction == -1):
            return self.r_detected
        else:
            return self.r_hidden
