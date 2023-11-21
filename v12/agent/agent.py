import numpy as np
import logging

from agent.abstract_agent import AbstractAgent
from agent.agent_representation import AgentRepresentation
from environment.state_handling import get_num_configs
from v12.agent.model import ModelQLearning
from config import config

SYSCALL_DIMS = config.get_int("v12", "syscall_dims")
HIDDEN_NEURONS = config.get_int("v12", "hidden_neurons")
LEARN_RATE = config.get_float("v12", "learn_rate")
DISCOUNT_FACTOR = config.get_float("v12", "discount_factor")


class SyscallAgentQLearning(AbstractAgent):
    def __init__(self, representation: AgentRepresentation = None):
        self.representation = representation

        if isinstance(representation, AgentRepresentation):  # build from representation
            self.num_input = representation.num_input
            self.num_hidden = representation.num_hidden
            self.num_output = representation.num_output
            self.actions = list(range(representation.num_output))

            self.learn_rate = representation.learn_rate
            self.model = ModelQLearning(learn_rate=LEARN_RATE, num_configs=self.num_output)
        else:  # init from scratch
            num_configs = get_num_configs()
            self.num_input = SYSCALL_DIMS  # Input size
            self.num_hidden = HIDDEN_NEURONS  # Hidden neurons
            self.num_output = num_configs  # Output size
            self.actions = list(range(num_configs))

            self.learn_rate = LEARN_RATE  # only used in AbstractAgent for storing AgentRepresentation
            self.model = ModelQLearning(learn_rate=LEARN_RATE, num_configs=num_configs)

    def initialize_network(self):
        logging.debug("AGENT: initializing network...")
        if isinstance(self.representation, AgentRepresentation):  # init from representation
            weights1 = np.asarray(self.representation.weights1)
            weights2 = np.asarray(self.representation.weights2)
            bias_weights1 = np.asarray(self.representation.bias_weights1)
            bias_weights2 = np.asarray(self.representation.bias_weights2)
        else:  # init from scratch
            # uniform weight initialization
            weights1 = np.random.uniform(0, 1, (self.num_input, self.num_hidden))
            weights2 = np.random.uniform(0, 1, (self.num_hidden, self.num_output))

            bias_weights1 = np.zeros((self.num_hidden, 1))
            bias_weights2 = np.zeros((self.num_output, 1))

        return weights1, weights2, bias_weights1, bias_weights2

    def predict(self, weights1, weights2, bias_weights1, bias_weights2, epsilon, state):
        logging.debug("AGENT: predict...")

        hidden, q_values, selected_action = self.model.forward(weights1, weights2, bias_weights1, bias_weights2,
                                                               epsilon, inputs=state)
        logging.debug(f"AGENT: done predicting: Selected action: {selected_action}")
        return hidden, q_values, selected_action

    def update_weights(self, q_values, error, state, hidden, weights1, weights2, bias_weights1, bias_weights2):
        logging.debug("AGENT: update weights...")
        new_w1, new_w2, new_bw1, new_bw2 = self.model.backward(q_values, error, hidden, weights1, weights2,
                                                               bias_weights1, bias_weights2, inputs=state)
        return new_w1, new_w2, new_bw1, new_bw2

    def update_error(self, error, reward, selected_action, curr_q_values, next_q_values, is_done):
        logging.debug("AGENT: update error...")
        if is_done:
            error[selected_action] = reward - curr_q_values[selected_action]
        else:
            # off-policy
            error[selected_action] = reward + (DISCOUNT_FACTOR * np.max(next_q_values)) - curr_q_values[selected_action]
        logging.debug(f"AGENT: err {error.T}")
        return error

    def init_error(self):
        return np.zeros((self.num_output, 1))


