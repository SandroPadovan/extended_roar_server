from time import sleep, time
from datetime import datetime
import logging

from agent.agent_representation import AgentRepresentation
from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward.syscall_standard_reward import SyscallStandardReward
from environment.state_handling import is_syscall_ready, set_syscall_ready, is_rw_done, is_simulation, get_prototype, \
    collect_syscall
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done
from v2.environment.plotting import plot_average_results
from config import config

WAIT_FOR_CONFIRM = config.get_default_bool('controller', 'wait_for_confirm', default=False)
MAX_STEPS_V12 = config.get_int("v12", "max_steps")
EPSILON = config.get_float("v12", "epsilon")
FEATURE_NAME = config.get("anomaly_detection", "syscall_feature")


class SyscallControllerQLearning(AbstractController):

    def run_c2(self):
        logging.info("\n==============================\nPrepare Reward Computation\n==============================")
        SyscallStandardReward.prepare_reward_computation()

        if WAIT_FOR_CONFIRM:
            cont = input("Results ok? Start C2 Server? [y/n]\n")
            if cont.lower() == "y":
                # use name mangling to access __start_training method
                super()._AbstractController__start_training()
        else:
            super()._AbstractController__start_training()

    def loop_episodes(self, agent):

        logging.debug("Controller: loop episodes...")

        start_timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        run_info = "p{}-{}s".format(get_prototype(), MAX_STEPS_V12)
        description = "{}={}".format(start_timestamp, run_info)

        # ==============================
        # Setup agent
        # ==============================

        weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()

        # ==============================
        # Setup environment
        # ==============================

        reward_system = SyscallStandardReward(+50, +20, -20)
        last_action = -1
        reward_store = []
        last_q_values = []

        sim_step = 1
        sim_start = time()

        # accept initial Syscall
        logging.debug("Wait for initial Syscall...")
        if is_simulation():
            simulate_sending_fp(0)
        while not is_syscall_ready():
            logging.debug("waiting until syscall is ready...")
            sleep(.5)
        curr_syscall = collect_syscall()
        set_syscall_ready(False)

        logging.debug("Loop episode...")
        while not is_rw_done():
            logging.info("==================================================")
            # ==============================
            # Predict action
            # ==============================

            state = curr_syscall

            curr_hidden, curr_q_values, selected_action = agent.predict(weights1, weights2, bias_weights1,
                                                                        bias_weights2, EPSILON, state=state)

            logging.info(f"Predicted action: {selected_action}. Step {sim_step}")

            # ==============================
            # Take step and observe new state
            # ==============================

            # convert action to rw_config and send to client
            if selected_action != last_action:
                logging.debug(f"Sending new action {selected_action} to client.")
                rw_config = map_to_ransomware_configuration(selected_action)
                if not is_simulation():  # cannot send if no socket listening during simulation
                    send_config(selected_action, rw_config)
            last_action = selected_action

            sim_step += 1

            # receive next FP and compute reward based on FP
            logging.debug("Wait for Syscall...")
            if is_simulation():
                simulate_sending_fp(selected_action)
            while not (is_syscall_ready() or is_rw_done()):
                sleep(.5)

            if is_rw_done():
                next_syscall = curr_syscall
            else:
                next_syscall = collect_syscall()

            next_state = next_syscall
            set_syscall_ready(False)

            # ==============================
            # Observe reward for new state
            # ==============================

            if is_simulation() and sim_step > MAX_STEPS_V12:
                simulate_sending_rw_done()

            logging.debug("Computing reward for next Syscall.")
            reward = reward_system.compute_reward(next_state.tolist(), is_rw_done())
            reward_store.append((selected_action, reward))

            # ==============================
            # Next Q-values, error, and learning
            # ==============================

            # initialize error
            error = agent.init_error()

            if is_rw_done():
                # update error based on observed reward
                error = agent.update_error(error, reward, selected_action, curr_q_values, next_q_values=None,
                                           is_done=True)

                # send error to agent, update weights accordingly
                weights1, weights2, bias_weights1, bias_weights2 = agent.update_weights(curr_q_values, error, state,
                                                                                        curr_hidden, weights1, weights2,
                                                                                        bias_weights1, bias_weights2)
                last_q_values = curr_q_values
            else:
                # predict next Q-values and action
                logging.debug("Predict next action.")
                next_hidden, next_q_values, next_action = agent.predict(weights1, weights2, bias_weights1,
                                                                        bias_weights2, EPSILON, state=next_state)
                logging.debug(f"Predicted next action {next_action}")

                # update error based on observed reward
                error = agent.update_error(error, reward, selected_action, curr_q_values, next_q_values, is_done=False)

                # send error to agent, update weights accordingly
                weights1, weights2, bias_weights1, bias_weights2 = agent.update_weights(curr_q_values, error, state,
                                                                                        curr_hidden, weights1, weights2,
                                                                                        bias_weights1, bias_weights2)

            # ==============================
            # Prepare next step
            # ==============================

            # update current state
            curr_syscall = next_syscall

            # ========== END OF STEP ==========

        # ========== END OF EPISODE ==========
        agent_file = AgentRepresentation.save_agent(weights1, weights2, bias_weights1, bias_weights2,
                                                    EPSILON, agent, description)

        sim_end = time()
        logging.info("Took: {}s, roughly {}min.".format("%.3f" % (sim_end - sim_start), "%.1f" % ((sim_end - sim_start)
                                                                                                  / 60)))

        logging.info("==============================")
        logging.info("Saving trained agent to file...")
        logging.info(f"- Agent saved: {agent_file}")

        all_rewards = list(map(lambda x: x[1], reward_store))
        logging.info("Generating plots...")
        results_plots_file = plot_average_results(all_rewards, MAX_STEPS_V12, description)
        logging.info(f"- Plots saved: {results_plots_file}")
        results_store_file = AbstractController.save_results_to_file(all_rewards, [], [MAX_STEPS_V12], description)
        logging.info(f"- Results saved: {results_store_file}")

        return last_q_values, reward_store
