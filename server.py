from argparse import ArgumentParser
from multiprocessing import Process
from time import sleep

from api import create_app
from environment.constructor import get_controller
from environment.state_handling import get_instance_number, setup_child_instance, initialize_storage, cleanup_storage,\
    is_multi_fp_collection, set_multi_fp_collection, is_simulation, set_simulation, set_api_running, set_prototype,\
    set_agent_representation_path
from config import config


def parse_args():
    parser = ArgumentParser(description='C2 Server')
    parser.add_argument('-c', '--collect',
                        help='Indicator to only collect incoming fingerprints instead of running the full C2 server.',
                        default=False,
                        action="store_true")
    parser.add_argument('-p', '--proto',
                        help='Prototype selection.',
                        default=0,
                        action="store")
    parser.add_argument('-r', '--representation',
                        help='Absolute path to agent representation file.',
                        default="",
                        action="store")
    parser.add_argument('-s', '--simulation',
                        help='Indicator for simulation of sensor behavior.',
                        default=False,
                        action="store_true")

    return parser.parse_args()


def start_api(instance_number):
    setup_child_instance(instance_number)
    app = create_app()
    print("==============================\nStart API\n==============================")
    set_api_running()
    host = config.get('server', 'host')
    port = config.get_int('server', 'port')
    app.run(host, port)


def kill_process(process):
    print("Kill Process", process)
    process.kill()
    process.join()


if __name__ == "__main__":
    procs = []
    try:
        print("==============================\nInstantiate Storage")
        initialize_storage()
        print("- Storage ready.")

        # Parse arguments
        args = parse_args()
        collect = args.collect
        set_multi_fp_collection(collect)
        proto = args.proto
        set_prototype(proto)
        simulated = args.simulation
        set_simulation(simulated)
        agent_repr = args.representation
        set_agent_representation_path(agent_repr)

        # Start API listener
        if not is_simulation():
            proc_api = Process(target=start_api, args=(get_instance_number(),))
            procs.append(proc_api)
            proc_api.start()

        # Start C2 server
        if not is_multi_fp_collection():
            controller = get_controller()
            controller.run_c2()
        else:
            while True:
                sleep(600)  # sleep until process is terminated by user keyboard interrupt
    finally:
        if is_multi_fp_collection():
            print("==============================")
        print("Final Cleanup")
        for proc in procs:
            kill_process(proc)
        print("- Parallel processes killed.")
        cleanup_storage()
        print("- Storage cleaned up.\n==============================")
