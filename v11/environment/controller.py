from time import sleep
import logging

from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward.syscall_standard_reward import SyscallStandardReward
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, is_simulation
from utilities.simulate import simulate_sending_fp
from config import config
from environment.anomaly_detection.syscalls.get_features import get_features
from environment.state_handling import get_syscall_file_path

WAIT_FOR_CONFIRM = config.get_default_bool('controller', 'wait_for_confirm', default=False)


class SyscallController(AbstractController):

    def run_c2(self):
        logging.info("==============================\nPrepare Reward Computation\n==============================")
        SyscallStandardReward.prepare_reward_computation()

        if WAIT_FOR_CONFIRM:
            cont = input("Results ok? Start C2 Server? [y/n]\n")
            if cont.lower() == "y":
                # use name mangling to access __start_training method
                super()._AbstractController__start_training()
        else:
            super()._AbstractController__start_training()

    def loop_episodes(self, agent):
        # setup
        reward_system = SyscallStandardReward(+1, 0, -1)
        last_action = None

        # accept initial Syscall
        logging.info("Wait for initial Syscall...")
        if is_simulation():
            simulate_sending_fp(0)
        while not is_fp_ready():
            sleep(.5)
        curr_syscall = get_features(raw_data_path=get_syscall_file_path(),
                                    vectorizers_path=config.get("anomaly_detection", "normal_vectorizers_path"))
        set_fp_ready(False)

        logging.debug("Loop episode...")
        while True:

            # agent selects action based on state
            logging.info("Predict next action.")
            selected_action, is_last = agent.predict(curr_syscall)
            logging.info("Predicted action {}; is last: {}.".format(selected_action, is_last))

            # convert action to rw_config and send to client
            if selected_action != last_action:
                logging.info("Sending new action {} to client.".format(selected_action))
                rw_config = map_to_ransomware_configuration(selected_action)
                if not is_simulation():  # cannot send if no socket listening during simulation
                    send_config(selected_action, rw_config)
            last_action = selected_action

            # receive next Syscall and compute reward based on Syscall
            logging.info("Wait for Syscall...")
            if is_simulation():
                simulate_sending_fp(selected_action)
            while not (is_fp_ready()):
                sleep(.5)

            next_syscall = get_features(raw_data_path=get_syscall_file_path(),
                                        vectorizers_path=config.get("anomaly_detection", "normal_vectorizers_path"))
            feature_name = config.get('anomaly_detection', 'syscall_feature')

            set_fp_ready(False)

            logging.info("Computing reward for next Syscall.")
            reward = reward_system.compute_reward(next_syscall[feature_name].tolist(), is_rw_done())
            logging.info(reward)

            if is_last:
                # terminate episode instantly
                logging.info("Terminate episode.")
                break
            # set next_syscall to curr_fp for next iteration
            curr_syscall = next_syscall

        return [], []
