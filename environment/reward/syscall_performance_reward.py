import math
import logging
import json

from environment.reward.abstract_reward import AbstractReward
from environment.anomaly_detection.syscall_anomaly_detection import train_syscall_anomaly_detection, \
    detect_syscall_anomaly
from environment.state_handling import collect_rate
from config import config


class SyscallPerformanceReward(AbstractReward):
    def __init__(self, r_done, r_hidden, r_detected):
        self.r_done = r_done
        self.r_hidden = r_hidden
        self.r_detected = r_detected
        self.use_ideal_AD = config.get_default_bool("anomaly_detection", "use_ideal_AD", False)

    @staticmethod
    def prepare_reward_computation():
        train_syscall_anomaly_detection()

    def compute_reward(self, syscall_features, action, done):
        rate = collect_rate()
        logging.debug(f"REWARD: rate {rate}")

        if self.use_ideal_AD:
            hidden_configs = json.loads(config.get("anomaly_detection", "hidden_configs"))
            anomalous = bool(action not in hidden_configs)
        else:
            anomalous = detect_syscall_anomaly(syscall_features)[0] == -1  # int [-1, 1]

        logging.debug("--- Detected {} Syscall.".format("anomalous" if anomalous else "normal"))

        if anomalous:
            logging.debug(f"REWARD: det {self.r_detected} {rate} {max(rate, 1)}")
            reward = -(max(1, abs(self.r_detected)) / max(rate, 1)) - abs(self.r_detected)  # -d/r - d
        elif done:
            reward = self.r_done
        else:
            logging.debug(f"REWARD: hid {rate} {100 * math.log(0.01*rate+1)} {self.r_hidden}")
            reward = 100 * math.log(0.01*rate + 1) + abs(self.r_hidden)  # 100 * ln(0.01*r + 1) + h
        logging.debug(f"REWARD: result {reward}")
        return round(reward, 5), anomalous
