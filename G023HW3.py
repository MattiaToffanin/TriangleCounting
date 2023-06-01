from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys

# Number of items to read
THRESHOLD = 10000000


def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, histogram
    batch_size = batch.count()
    streamLength[0] += batch_size
    # Extract the distinct items from the batch
    batch_items = batch.map(lambda s: (int(s), 1)).reduceByKey(lambda i1, i2: 1).collectAsMap()

    # Update the streaming state
    for key in batch_items:
        if key not in histogram:
            histogram[key] = 1

    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

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

    # Print parameters info
    print("Number of rows of the count sketch =", D)
    print("Number of columns of the count sketch =", W)
    print("Left endpoint of the interval of interest =", left)
    print("Right endpoint of the interval of interest =", right)
    print("Number of top frequent items of interest =", K)
    print("Port number =", portExp)

    # Data structures to maintain the state of the stream
    streamLength = [0]
    histogram = {}

    # Stream creation
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)

    # Stream reading
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    ssc.stop(False, True)
    print("Streaming engine stopped")

    # Print output
    print("Number of items processed =", streamLength[0])
    print("Number of distinct items =", len(histogram))
    largest_item = max(histogram.keys())
    print("Largest item =", largest_item)
