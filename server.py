from argparse import ArgumentParser

from api import create_app
from environment.controller import run_c2
from environment.state_handling import is_multi_fp_collection, set_multi_fp_collection, initialize_storage


def parse_args():
    parser = ArgumentParser(description='C2 Server')
    parser.add_argument('-c', '--collect',
                        help='Indicator to only collect incoming fingerprints instead of running the full C2 server.',
                        action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    print("==============================\nInstantiate Storage\n")
    initialize_storage()

    # Parse arguments
    args = parse_args()
    collect = args.collect or False
    set_multi_fp_collection(collect)

    run_c2()

    # Start API
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
