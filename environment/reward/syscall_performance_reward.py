import math
import logging

from environment.reward.abstract_reward import AbstractReward
from environment.anomaly_detection.syscall_anomaly_detection import train_syscall_anomaly_detection, \
    detect_syscall_anomaly
from environment.state_handling import collect_rate


class SyscallPerformanceReward(AbstractReward):
    def __init__(self, r_done, r_hidden, r_detected):
        self.r_done = r_done
        self.r_hidden = r_hidden
        self.r_detected = r_detected

    @staticmethod
    def prepare_reward_computation():
        train_syscall_anomaly_detection()

    def compute_reward(self, syscall_features, done):
        rate = collect_rate()
        logging.debug(f"REWARD: rate {rate}")

        anomalous = detect_syscall_anomaly(syscall_features)[0] == -1  # int [-1, 1]
        logging.info("--- Detected {} Syscall.".format("anomalous" if anomalous else "normal"))

        if anomalous:
            logging.debug(f"REWARD: det {self.r_detected} {rate} {max(rate, 1)}")
            reward = -(max(1, abs(self.r_detected)) / max(rate, 1)) - abs(self.r_detected)  # -d/r - d
        elif done:
            reward = self.r_done
        else:
            logging.debug(f"REWARD: hid {rate} {10 * math.log(rate+1)} {self.r_hidden}")
            reward = 10 * math.log(rate + 1) + abs(self.r_hidden)  # ln(r+1) + h
        logging.debug(f"REWARD: result {reward}")
        return round(reward, 5), anomalous
