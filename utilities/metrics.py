import os
import re
from datetime import datetime
from threading import Lock

from environment.state_handling import set_fp_ready, get_fp_file_path, get_rate_file_path, \
    get_storage_path


def write_resource_metrics_to_file(rate, fp, storage_path, is_multi):
    # print("UTILS: writing rate/fp", rate, fp, is_multi)
    if not is_multi:
        __write_rate_to_file(rate, storage_path)
    __write_fingerprint_to_file(fp, storage_path, is_multi)
    # print("UTILS: rate/fp written")
    set_fp_ready(True)


def write_syscall_metrics_to_file(raw_data_file, storage_path):

    def extract_metrics(line: str):
        line = re.split(r' |\( |\)', line)
        line = list(filter(lambda a: a != '', line))
        if len(line) < 6:
            return None
        timestamp = line[0]
        time_cost = line[1]
        pid = line[4]
        if line[5] == '...':
            if timestamp == '0.000':
                return None
            else:
                syscall = line[7].split('(')[0]
        else:
            syscall = line[5].split('(')[0]
        return [pid, timestamp, syscall, time_cost]

    # preprocess raw data
    raw_data_file.stream.seek(0)
    lines = raw_data_file.stream.readlines()

    os.makedirs(storage_path, exist_ok=True)
    file_name = "sc-{time}.csv".format(time=datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    sc_path = os.path.join(storage_path, file_name)

    with open(sc_path, 'w') as outp:

        # write headers
        outp.write('pid,timestamp,syscall,time_cost\n')

        for line in lines:
            try:
                res = extract_metrics(line.decode("utf-8"))
            except Exception as e:
                print(e)
                res = None
            if res is not None:
                [pid, timestamp, syscall, time_cost] = res
                outp.write('{},{},{},{}\n'.format(pid, timestamp, syscall, time_cost))

        print(f'Successfully preprocessed incoming raw syscall data. Stored to: {file_name}')
        outp.close()


def __write_rate_to_file(rate, storage_path):
    os.makedirs(storage_path, exist_ok=True)
    rate_path = get_rate_file_path()

    lock = Lock()
    with lock:
        with open(rate_path, "w") as file:
            file.write(str(rate))


def __write_fingerprint_to_file(fp, storage_path, is_multi):
    os.makedirs(storage_path, exist_ok=True)
    if is_multi:
        file_name = "fp-{time}.txt".format(time=datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        fp_path = os.path.join(storage_path, file_name)
    else:
        fp_path = get_fp_file_path()

    lock = Lock()
    with lock:
        with open(fp_path, "x" if is_multi else "w") as file:
            file.write(fp)
