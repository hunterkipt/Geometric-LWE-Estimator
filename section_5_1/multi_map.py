from time import process_time
from pathos.multiprocessing import Pool
import numpy as np

def test(id, *args):
    return id * 10

def target_wrapper(target, id, args):
    try:
        start = process_time()
        res = target(id, args)
        end = process_time()
    except Exception as e:
        print(type(e))
        print(e)
        return None

    return end - start, res

def multi_map(num_trials, num_threads, target, args):
    pool = Pool(processes=num_threads)
    results = []
    i = 0
    while len(results) < num_trials:
        thread_outs = [pool.apply_async(target_wrapper, (target, j, args)) for j in range(i*num_threads, (i+1)*num_threads)]
        result = [thread_out.get() for thread_out in thread_outs]
        results += list(filter(None, result))
        i += 1

    runtimes, results = zip(*results[:num_trials])
    avg_runtime = np.mean(runtimes)

    pool.terminate()
    pool.join()

    return avg_runtime, results