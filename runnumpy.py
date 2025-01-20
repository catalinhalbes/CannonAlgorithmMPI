from runutils import *
from time import perf_counter
import numpy as np
import sys
import os

RERUNS = 20
RESULTS_PATH = "./result"
CSV_FILE = "./results_numpy.csv"

class Metrics:
    exe_name = ""
    threads = 0
    runs = 0
    total_time = 0.0
    read_time = 0.0
    write_time = 0.0
    dot_time = 0.0
    errors = 0
    mape = 0.0
    mae = 0.0

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} [matrix_1] [matrix_2] [expected_res]")
        exit(1)
        
    input_matrices = sys.argv[1:3]
    expected_result = sys.argv[3]

    # run step
    result = RESULTS_PATH + ".bin"
    metrics = Metrics()
    metrics.name = "mulmat.py"

    for run_idx in range(RERUNS):
        print(f"{run_idx} ", end="", flush=True)
        start = perf_counter()

        proc = Popen(["py", "mulmat.py", input_matrices[0], input_matrices[1], result], stdout=PIPE)
        (out, err) = proc.communicate()
        exit_code = proc.wait()

        stop = perf_counter()

        if exit_code != 0:
            print(f"{exe} failed at run {run_idx}")
            results[exe].errors += 1
            continue

        if err:
            print(f"{exe} returned an error at {run_idx}: \n{prepare_out(err)}")
            results[exe].errors += 1
            continue

        times = out.decode("utf-8").strip().replace("\\r", "").split("\n")

        if len(times) != 3:
            print(f"{exe} returned an invalid output at {run_idx}: \n{times}")
            results[exe].errors += 1
            continue

        metrics.runs += 1
        metrics.total_time += (stop - start) * 1000
        metrics.read_time += float(times[0])
        metrics.dot_time += float(times[1])
        metrics.write_time += float(times[2])
        metrics.mape += mat_MAPE(result, expected_result)
        metrics.mae += mat_MAE(result, expected_result)

    print()

    # print results
    res = metrics
    print(f"\n============================================\nTest: {res.name}_{res.threads}")
    print(f"Successfull runs: {res.runs}")
    print(f"Errors: {res.errors}")
    print(f"Average run time: {res.total_time / res.runs}")
    print(f"Average read time: {res.read_time / res.runs}")
    print(f"Average dot time: {res.dot_time / res.runs}")
    print(f"Average write time: {res.write_time / res.runs}")
    print(f"Average MAPE: {res.mape / res.runs}")
    print(f"Average MAE: {res.mae / res.runs}")

    # save to csv
    with open(CSV_FILE, "w") as csv_file:
        csv_file.write("Executable,Avg_Total_Time,Avg_Dot_Time\n")
        tot_time = res.total_time/res.runs
        dot_time = res.dot_time/res.runs
        csv_file.write(f"{res.name},{tot_time},{dot_time}\n")
