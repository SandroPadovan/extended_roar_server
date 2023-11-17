from agent.abstract_agent import AbstractAgent
from environment.state_handling import get_num_configs


class SyscallManualAgent(AbstractAgent):
    def __init__(self):
        self.next_action = 0
        self.num_actions = get_num_configs()

    def predict(self, fingerprint):
        next_a = self.next_action
        self.next_action += 1
        # we return the last valid action; next_action zero-based, num_actions one-based
        is_last = self.next_action >= self.num_actions
        return next_a, is_last

    def update_weights(self, fingerprint, reward):
        # not required in version 11
        pass
