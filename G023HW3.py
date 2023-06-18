from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import random as rand
import numpy as np

THRESHOLD = 10000000  # Number of items to read

p = 8191  # Prime number
a = rand.randint(1, p - 1)  # Random number at each run
b = rand.randint(1, p - 1)  # Random number at each run
c = rand.randint(1, p - 1)  # Random number at each run
d = rand.randint(1, p - 1)  # Random number at each run


# Hash function
def h(u, j):
    rand.seed((a * u + b * j))
    return rand.randint(1, p) % W


# Hash function
def g(u, j):
    rand.seed((c * u + d * j))
    val = rand.randint(0, 1)
    return -1 if val == 0 else +1


def process_batch(time, batch):
    global streamLength, true_frequencies, C, left, right, D, W
    batch_size = batch.count()
    # If we already have enough points (> THRESHOLD), skip this batch.
    if streamLength[0] >= THRESHOLD:
        return
    streamLength[0] += batch_size  # Increment the entire stream length
    # Extract the distinct items from the batch in [left; right]
    batch_items = batch.map(lambda s: int(s)).filter(lambda x: left <= x <= right).collect()[:THRESHOLD]
    streamLength[1] += len(batch_items)  # Increment the number of items processed

    for xt in batch_items:
        for j in range(D):
            C[j, h(xt, j)] += g(xt, j)

        if xt not in true_frequencies:
            true_frequencies[xt] = 1
        else:
            true_frequencies[xt] += 1

    # Stopping condition
    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()


if __name__ == '__main__':
    # Checking number of command line parameters
    assert len(sys.argv) == 7, "Usage: python G023HW3.py <D> <W> <left> <right> <K> <portExp>"

    # SPARK setup
    conf = SparkConf().setMaster("local[*]").setAppName("CountSketch")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 1)  # Batch duration of 1 second
    ssc.sparkContext.setLogLevel("ERROR")

    # Semaphore
    stopping_condition = threading.Event()

    # Read D parameter
    D = sys.argv[1]
    assert D.isdigit(), "D must be an integer"
    D = int(D)

    # Read W parameter
    W = sys.argv[2]
    assert W.isdigit(), "W must be an integer"
    W = int(W)

    # Read left parameter
    left = sys.argv[3]
    assert left.isdigit(), "left must be an integer"
    left = int(left)

    # Read right parameter
    right = sys.argv[4]
    assert right.isdigit(), "right must be an integer"
    right = int(right)

    # Read K parameter
    K = sys.argv[5]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # Read portExp parameter
    portExp = sys.argv[6]
    assert portExp.isdigit(), "portExp must be an integer"
    portExp = int(portExp)

    # Data structures to maintain the state of the stream
    streamLength = [0, 0]  # i0: entire stream length, i1: filtered stream length
    true_frequencies = {}  # dictionary to store the true frequencies
    C = np.zeros((D, W))  # matrix to calculate the approximate frequencies

    # Stream creation
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)

    # Stream reading
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    # Starting streaming engine
    ssc.start()
    # Waiting for shutdown condition
    stopping_condition.wait()
    # Stopping the streaming engine
    ssc.stop(False, True)
    # Streaming engine stopped

    # Approximate frequency calculation
    appr_frequencies = {}  # dictionary to store the approximate frequencies
    for item in range(left, right + 1):
        appr_frequencies[item] = int(np.median([g(item, j) * C[j, h(item, j)] for j in range(D)]))

    # Approximate second moment calculation
    f2j = np.zeros(D)  # list to store intermediate approximate second moments
    for j in range(D):
        for k in range(W):
            f2j[j] += C[j, k] ** 2
    appr_second_moment = np.median(f2j)
    appr_second_moment /= streamLength[1] ** 2

    # True second moment calculation
    true_second_moment = 0
    for k in true_frequencies:
        true_second_moment += true_frequencies[k] ** 2
    true_second_moment /= streamLength[1] ** 2

    # Order by value (frequencies) true_frequencies dictionary
    true_frequencies = dict(sorted(true_frequencies.items(), key=lambda it: it[1]))

    # Create a list with the K most frequent elements
    true_frequencies_list = list(true_frequencies.items())[-K:]

    # Average relative error calculation
    cumulative = 0
    for (item, freq) in true_frequencies_list:
        cumulative = cumulative + (abs(freq - appr_frequencies[item]) / freq)
    average_relative_error = cumulative / K

    # Print output
    print("D = %d W = %d [left,right] = [%d,%d] K = %d Port = %d" % (D, W, left, right, K, portExp))
    print("Total number of items =", streamLength[0])
    print("Total number of items in [%d,%d] =" % (left, right), streamLength[1])
    print("Number of distinct items in [%d,%d] =" % (left, right), len(appr_frequencies))
    if K <= 20:
        for (item, freq) in true_frequencies_list:
            print("Item %d Freq = %d Est. Freq = %d" % (item, freq, appr_frequencies[item]))
    print("Avg err for top %d =" % K, average_relative_error)
    print("F2 %f F2 Estimate %f" % (true_second_moment, appr_second_moment))
