from time import sleep, time
from datetime import datetime
import logging

from agent.agent_representation import AgentRepresentation
from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward.syscall_performance_reward import SyscallPerformanceReward
from environment.reward.revised_syscall_performance_reward import RevisedSyscallPerformanceReward
from environment.state_handling import is_syscall_ready, set_syscall_ready, is_rw_done, is_simulation, get_prototype, \
    collect_syscall, set_rw_done, collect_rate
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done
from utilities.plots import plot_average_results
from config import config

WAIT_FOR_CONFIRM = config.get_default_bool('controller', 'wait_for_confirm', default=False)
MAX_EPISODES_V13 = config.get_int("v13", "max_episodes")
USE_SIMULATED_CORPUS = config.get_default_bool("v13", "use_simulated_corpus", default=True)
MAX_STEPS_V13 = config.get_int("v13", "max_steps")
CORPUS_SIZE = config.get_int("v13", "corpus_size")
EPSILON = config.get_float("v13", "epsilon")
DECAY_RATE = config.get_float("v13", "decay_rate")
FEATURE_NAME = config.get("anomaly_detection", "syscall_feature")
USE_REVISED_REWARD = config.get_default_bool("v13", "use_revised_reward", default=False)


class SyscallControllerAdvancedQLearning(AbstractController):

    def run_c2(self):
        logging.info("\n==============================\nPrepare Reward Computation\n==============================")
        if USE_REVISED_REWARD:
            RevisedSyscallPerformanceReward.prepare_reward_computation()
        else:
            SyscallPerformanceReward.prepare_reward_computation()

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
        run_info = "p{}-{}e-{}s".format(get_prototype(), MAX_EPISODES_V13, CORPUS_SIZE) if USE_SIMULATED_CORPUS \
            else "p{}-{}s".format(get_prototype(), MAX_STEPS_V13)
        description = "{}={}".format(start_timestamp, run_info)
        agent_file = None

        # ==============================
        # Setup agent
        # ==============================

        weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()

        reward_system = RevisedSyscallPerformanceReward(+1000, 50, 50) if USE_REVISED_REWARD \
            else SyscallPerformanceReward(+1000, 0, 20)

        # ==============================
        # Setup collectibles
        # ==============================

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []

        last_q_values = []
        num_total_steps = 0
        all_start = time()

        eps_iter = range(1, MAX_EPISODES_V13 + 1)
        for episode in eps_iter:
            # ==============================
            # Setup environment
            # ==============================
            set_rw_done(False)

            # decay epsilon, episode 1-based
            epsilon_episode = EPSILON / (1 + DECAY_RATE * (episode - 1))

            last_action = -1
            reward_store = []
            summed_reward = 0

            steps = 0
            sim_encryption_progress = 0
            sim_step = 1
            eps_start = time()

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
                logging.debug("==================================================")
                # ==============================
                # Predict action
                # ==============================

                state = curr_syscall

                curr_hidden, curr_q_values, selected_action = agent.predict(weights1, weights2, bias_weights1,
                                                                            bias_weights2, epsilon_episode, state=state)

                logging.info(f"Step {sim_step}: Predicted action: {selected_action}")
                steps += 1
                sim_step += 1
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

                # receive next Syscall and compute reward based on Syscall
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

                # compute encryption progress (assume 1s per step)
                rate = collect_rate()
                sim_encryption_progress += rate

                # ==============================
                # Observe reward for new state
                # ==============================

                limit_reached = sim_encryption_progress >= CORPUS_SIZE if USE_SIMULATED_CORPUS \
                    else sim_step > MAX_STEPS_V13

                if is_simulation() and limit_reached:
                    logging.info("\nRW done!\n")
                    simulate_sending_rw_done()

                logging.debug("Computing reward for next Syscall.")
                reward, detected = reward_system.compute_reward(next_state.tolist(), selected_action, is_rw_done())
                reward_store.append((selected_action, reward))
                summed_reward += reward
                if detected:
                    set_rw_done()  # terminate episode

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
                                                                                            curr_hidden, weights1,
                                                                                            weights2, bias_weights1,
                                                                                            bias_weights2)
                    logging.info(f"Episode {episode} Q-Values:\n {curr_q_values}")
                    last_q_values = curr_q_values
                else:
                    # predict next Q-values and action
                    logging.debug("Predict next action.")
                    next_hidden, next_q_values, next_action = agent.predict(weights1, weights2, bias_weights1,
                                                                            bias_weights2, epsilon_episode,
                                                                            state=next_state)
                    logging.debug(f"Predicted next action {next_action}")

                    # update error based on observed reward
                    error = agent.update_error(error, reward, selected_action, curr_q_values, next_q_values,
                                               is_done=False)

                    # send error to agent, update weights accordingly
                    weights1, weights2, bias_weights1, bias_weights2 = agent.update_weights(curr_q_values, error, state,
                                                                                            curr_hidden, weights1,
                                                                                            weights2, bias_weights1,
                                                                                            bias_weights2)

                # ==============================
                # Prepare next step
                # ==============================

                # update current state
                curr_syscall = next_syscall

                # ========== END OF STEP ==========

            # ========== END OF EPISODE ==========
            eps_end = time()
            logging.info("Episode {} took: {}s, roughly {}min.".format(episode, "%.3f" % (eps_end - eps_start),
                                                                       "%.1f" % ((eps_end - eps_start) / 60)))
            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_avg_rewards.append(summed_reward / steps)  # average reward over episode
            all_num_steps.append(steps)

            agent_file = AgentRepresentation.save_agent(weights1, weights2, bias_weights1, bias_weights2,
                                                        epsilon_episode, agent, description)

        # ========== END OF TRAINING ==========
        all_end = time()
        logging.info("All episodes took: {}s, roughly {}min.".format("%.3f" % (all_end - all_start),
                                                                     "%.1f" % ((all_end - all_start) / 60)))
        logging.info(f"steps total {num_total_steps}, avg {num_total_steps / MAX_EPISODES_V13}")

        logging.info("==============================")
        logging.info("Saving trained agent to file...")
        logging.info(f"- Agent saved: {agent_file}")

        logging.info("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, MAX_EPISODES_V13,
                                                  description)
        logging.info(f"- Plots saved: {results_plots_file}")
        results_store_file = AbstractController.save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps,
                                                                     description)
        logging.info(f"- Results saved: {results_store_file}")
        return last_q_values, all_rewards
