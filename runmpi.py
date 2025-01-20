from runutils import *
from time import perf_counter
import numpy as np
import sys
import os

import datetime

RERUNS = 20
RESULTS_PATH = "./result"
CSV_FILE = f"./results{datetime.datetime.now()}.csv"

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
    progs = ["mpimatmul.c", "mpimatmul-openmp.c"]
    threads_tests = {
        BUILD_PATH + "mpimatmul": [(1, 1), (4, 1), (16, 1)],
        BUILD_PATH + "mpimatmul-openmp": [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (4, 1), (4, 2), (4, 4), (4, 8), (16, 1), (16, 2)],
    }
    execs = []
    results: list[Metrics] = []

    #compile step
    for filename in progs:
        (file, ext) = os.path.splitext(filename)
        if ext == ".c":
            print(f"Compiling {filename}...")
            if compile_c_mpi(file) != 0:
                print("Encountered error compiling! Exiting...")
                exit(0)
            execs.append(BUILD_PATH + file)
        else:
            execs.append(filename)

    # run step
    for exe in execs:
        for thr_n in threads_tests[exe]:

            result = RESULTS_PATH + ".bin"
            metrics = Metrics()
            metrics.name = exe
            metrics.threads = thr_n

            print(f"Running tests for {exe} with {thr_n} threads: ", end="", flush=True)

            for run_idx in range(RERUNS):
                print(f"{run_idx} ", end="", flush=True)
                start = perf_counter()

                proc = Popen(['mpirun', '--hostfile', 'hostfile.txt', '-n', str(thr_n[0]), exe, input_matrices[0], input_matrices[1], result, str(thr_n[1])], stdout=PIPE)
                (out, err) = proc.communicate()
                exit_code = proc.wait()

                stop = perf_counter()

                if exit_code != 0:
                    print(f"{exe} failed at run {run_idx}\nout: {out}\nerr: {err}")
                    metrics.errors += 1
                    continue

                if err:
                    print(f"{exe} returned an error at {run_idx}: \n{prepare_out(err)}")
                    metrics.errors += 1
                    continue

                times = out.decode("utf-8").strip().replace("\\r", "").split("\n")

                if len(times) != 3:
                    print(f"{exe} returned an invalid output at {run_idx}: \n{times}")
                    metrics.errors += 1
                    continue

                metrics.runs += 1
                metrics.total_time += (stop - start) * 1000
                metrics.read_time += float(times[0])
                metrics.dot_time += float(times[1])
                metrics.write_time += float(times[2])
                metrics.mape += mat_MAPE(result, expected_result)
                metrics.mae += mat_MAE(result, expected_result)

            results.append(metrics)
            print()

    # print results
    for res in results:
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
        csv_file.write("Executable,Thread_Config,Number_of_threads,Avg_Total_Time,Avg_Dot_Time,Total_Speedup,Dot_Speedup,Total_Efficiency,Dot_Efficency,Tot_Cost,Dot_Cost\n")
        t_secv = results[0].total_time / results[0].runs
        dot_secv = results[0].dot_time / results[0].runs
        for res in results:
            thread_config = str(res.threads)
            thread_num = np.prod(res.threads[0] * res.threads[1])
            tot_time = res.total_time/res.runs
            dot_time = res.dot_time/res.runs
            csv_file.write(f"{res.name},{thread_config},{thread_num},{tot_time},{dot_time},{t_secv/tot_time},{dot_secv/dot_time},{t_secv/tot_time/thread_num},{dot_secv/dot_time/thread_num},{tot_time*thread_num},{dot_time*thread_num}\n")
