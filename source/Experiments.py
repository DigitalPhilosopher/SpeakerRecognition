import os
import sys
import threading
import multiprocessing
import subprocess
import argparse
from utils import list_cuda_devices
import time


def main(args):
    get_commands(args)

    print(f"Start training with {len(trainCommands)} commands.")
    run_commands(trainCommands)

    print(f"Start inference with {len(inferenceCommands)} commands.")
    run_commands(inferenceCommands)

    print(f"Start analytics with {len(analyticsCommands)} commands.")
    run_commands(analyticsCommands)


def get_commands(args):
    global trainCommands, analyticsCommands, inferenceCommands

    experiments = process_file(args.experiments)

    trainCommands, analyticsCommands, inferenceCommands = [], [], []
    for experiment in experiments:
        command = experiment.split(" ")
        if "source/TrainModel.py" in experiment:
            trainCommands.append(command)
        elif "source/Analytics.py" in experiment:
            analyticsCommands.append(command)
        elif "source/Inference.py" in experiment:
            inferenceCommands.append(command)


def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    experiments = [line.strip() for line in lines if line.strip()
                   and not line.strip().startswith('#')]

    return experiments


def run_commands(commands):
    devices = list_cuda_devices()

    list_lock = threading.Lock()

    manager = multiprocessing.Manager()
    shared_commands = manager.list(commands)

    # Create threads
    threads = []
    for device in devices:
        thread = threading.Thread(
            target=thread_function, args=(device, shared_commands, list_lock))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


def thread_function(name, commands, list_lock):
    while commands:
        # Lock to ensure only one thread modifies the list at a time
        with list_lock:
            if commands:
                command = commands.pop(0)
            else:
                return
        command.append("--device")
        command.append(name)
        with open(f'logs/{name}.log', 'a') as log_file:
            pcommand = ' '.join(command)
            print(
                f"Starting new process on device {name}:\n\t$ {pcommand}")
            log_file.write(
                f"Starting new process on device {name}:\n\t$ {pcommand}")

            start_time = time.time()

            # Start the process and wait for it to finish
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Get the output and errors
            stdout, stderr = process.communicate()

            # Write the output and errors to the log file
            log_file.write("Output:\n")
            log_file.write(stdout.decode())

            log_file.write("\nErrors:\n")
            log_file.write(stderr.decode())

            # Check the return code and write it to the log file
            if process.returncode == 0:
                log_file.write("\nProcess finished successfully\n")
            else:
                log_file.write("\nProcess finished with errors\n")

            end_time = time.time()
            elapsed_time = end_time - start_time

            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)

            print(f"Finished process on device {name}:\n\t$ {pcommand}")
            print(
                f"\t-> Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running several experiments on ECAPA-TDNN Model for Deepfake Speaker Verification"
    )

    parser.add_argument(
        "--experiments",
        type=str,
        required=False,
        default="experiments.txt",
        help="File of experiments to run"
    )

    args = parser.parse_args()

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    sys.exit(main(args))
