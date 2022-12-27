import math
import os

import numpy as np
import pandas as pd

from agent.abstract_agent import AbstractAgent
from environment.anomaly_detection.constructor import get_preprocessor
from environment.settings import ALL_CSV_HEADERS, CSV_FOLDER_PATH
from environment.state_handling import get_num_configs
from v3.agent.model import ModelAdvancedQLearning

LEARN_RATE = 0.0025
DISCOUNT_FACTOR = 0.75


class AgentAdvancedQLearning(AbstractAgent):
    def __init__(self):
        num_configs = get_num_configs()
        self.actions = list(range(num_configs))

        self.fp_features = self.__get_fp_features()

        self.num_input = len(self.fp_features)  # Input size
        self.num_hidden = math.ceil(
            round(self.num_input / 2 / 10) * 10)  # Hidden neurons, next 10 from half the input size
        self.num_output = num_configs  # Output size

        self.learn_rate = LEARN_RATE
        self.model = ModelAdvancedQLearning(learn_rate=self.learn_rate, num_configs=num_configs)

    def __get_fp_features(self):
        df_normal = pd.read_csv(os.path.join(CSV_FOLDER_PATH, "normal-behavior.csv"))
        preprocessor = get_preprocessor()
        ready_dataset = preprocessor.preprocess_dataset(df_normal)
        return ready_dataset.columns

    def __preprocess_fp(self, fp):
        headers = ALL_CSV_HEADERS.split(",")
        indexes = []
        for header in self.fp_features:
            indexes.append(headers.index(header))
        return fp[indexes]

    def initialize_network(self):
        # Xavier weight initialization
        weights1 = np.random.uniform(-1 / np.sqrt(self.num_input), +1 / np.sqrt(self.num_input),
                                     (self.num_input, self.num_hidden))
        weights2 = np.random.uniform(-1 / np.sqrt(self.num_hidden), +1 / np.sqrt(self.num_hidden),
                                     (self.num_hidden, self.num_output))

        bias_weights1 = np.zeros((self.num_hidden, 1))
        bias_weights2 = np.zeros((self.num_output, 1))

        return weights1, weights2, bias_weights1, bias_weights2

    def predict(self, weights1, weights2, bias_weights1, bias_weights2, epsilon, state):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        hidden, q_values, selected_action = self.model.forward(weights1, weights2, bias_weights1, bias_weights2,
                                                               epsilon, inputs=ready_fp)
        return hidden, q_values, selected_action

    def update_weights(self, q_values, error, state, hidden, weights1, weights2, bias_weights1, bias_weights2):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        new_w1, new_w2, new_bw1, new_bw2 = self.model.backward(q_values, error, hidden, weights1, weights2,
                                                               bias_weights1, bias_weights2, inputs=ready_fp)
        return new_w1, new_w2, new_bw1, new_bw2

    def init_error(self):
        return np.zeros((self.num_output, 1))

    def update_error(self, error, reward, selected_action, curr_q_values, next_q_values, is_done):
        # print("AGENT: R sel selval best bestval", reward, selected_action, curr_q_values, next_q_values)
        if is_done:
            error[selected_action] = reward - curr_q_values[selected_action]
        else:
            # off-policy
            error[selected_action] = reward + (DISCOUNT_FACTOR * np.max(next_q_values)) - curr_q_values[selected_action]
        # print("AGENT: err\n", error.T)
        return error
