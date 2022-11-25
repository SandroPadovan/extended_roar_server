from time import sleep

from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward import compute_reward
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation
from simulate import simulate_sending_fp, simulate_sending_rw_done


class Controller2(AbstractController):
    def loop_episodes(self, agent):
        # accept initial FP
        print("Wait for initial FP...")
        if is_simulation():
            simulate_sending_fp(0)
        while not is_fp_ready():
            sleep(.5)
        curr_fp = collect_fingerprint()
        set_fp_ready(False)

        last_action = -1
        reward_store = []

        sim_steps = 0

        print("Loop episode...")
        while not is_rw_done():
            # ==============================
            # Predict action
            # ==============================

            # transform FP into np array
            state = AbstractController.transform_fp(curr_fp)

            # agent selects action based on state
            print("Predict action.")
            selected_action, _, q_values = agent.predict(state)
            print("Predicted action", selected_action)

            # ==============================
            # Take step and observe new state
            # ==============================

            # convert action to config and send to client
            if selected_action != last_action:
                print("Sending new action {} to client.".format(selected_action))
                config = map_to_ransomware_configuration(selected_action)
                if not is_simulation():  # cannot send if no socket listening during simulation
                    send_config(config)
            last_action = selected_action

            sim_steps += 1

            # receive next FP and compute reward based on FP
            print("Wait for FP...")
            if is_simulation():
                simulate_sending_fp(selected_action)
            while not (is_fp_ready() or is_rw_done()):
                sleep(.5)

            if is_rw_done():
                next_fp = curr_fp
            else:
                next_fp = collect_fingerprint()

            next_state = AbstractController.transform_fp(next_fp)
            set_fp_ready(False)

            # ==============================
            # Observe reward for new state
            # ==============================

            print("Computing reward for next FP.")
            reward = compute_reward(next_state, is_rw_done())
            reward_store.append(reward)

            # ==============================
            # Next Q-values, error, and learning
            # ==============================

            if is_simulation() and sim_steps >= 50:
                simulate_sending_rw_done()

            # initialize error
            error = agent.init_error()

            if is_rw_done():
                # update error based on observed reward
                error = agent.update_error(error=error, reward=reward, is_done=True,
                                           selected_action=selected_action, selected_q_value=q_values[selected_action],
                                           best_next_action=None, best_next_q_value=None)

                # send error to agent, update weights accordingly
                agent.update_weights(state, error)
            else:
                # predict next Q-values
                print("Predict next action.")
                next_selected_action, best_next_action, next_q_values = agent.predict(next_state)
                print("Predicted next action", next_selected_action)

                # update error based on observed reward
                error = agent.update_error(error=error, reward=reward, is_done=False,
                                           selected_action=selected_action, selected_q_value=q_values[selected_action],
                                           best_next_action=best_next_action, best_next_q_value=next_q_values[best_next_action])

                # send error to agent, update weights accordingly
                agent.update_weights(state, error)

            # ==============================
            # Prepare next step
            # ==============================

            # update current state
            curr_fp = next_fp
            print("==================================================")

        return reward_store