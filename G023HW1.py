from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import numpy as np
import time


def MR_ApproxTCwithNodeColors(edges, C):
    return 1  # TO DO: implement Algorithm1


def MR_ApproxTCwithSparkPartitions(edges):
    return 2  # TO DO: implement Algorithm2


def main():
    # Checking number of command line parameters
    assert len(sys.argv) == 4, "Usage: python G023HW1.py <C> <R> <file_name>"

    # SPARK setup
    conf = SparkConf().setAppName('TriangleCounting')
    sc = SparkContext(conf=conf)

    # Read C parameter
    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)

    # Read R parameter
    R = sys.argv[2]
    assert R.isdigit(), "R must be an integer"
    R = int(R)

    # Read input file and subdivide it into K random partitions
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path, minPartitions=C)
    edges = rawData.map(lambda x: (int(x.split(',')[0]), int(x.split(',')[1])))
    edges = edges.repartition(C).cache()

    # Print parameters info
    print("Dataset =", data_path)
    print("Number of Edges =", edges.count())
    print("Number of Colors =", C)
    print("Number of Repetitions =", R)

    triangles_counter = []
    cumulative_running_time = 0
    for i in range(R):
        start_time = time.time()
        triangles_number = MR_ApproxTCwithNodeColors(edges, C)
        end_time = time.time()
        triangles_counter.append(triangles_number)
        cumulative_running_time += (end_time - start_time)
    median_triangles = np.median(triangles_counter)  # TO DO: verify if we can use numpy
    mean_running_time = cumulative_running_time / R

    # Print Algorithm1 output
    print("Approximation through node coloring")
    print("- Number of triangles (median over", R, "runs) = %d" % median_triangles)
    print("- Running time (average over", R, "runs) = %d" % (mean_running_time * 1000), "ms")

    start_time = time.time()
    triangles_number = MR_ApproxTCwithSparkPartitions(edges)
    end_time = time.time()
    running_time = end_time - start_time

    # Print Algorithm2 output
    print("Approximation through Spark partitions")
    print("- Number of triangles = %d" % triangles_number)
    print("- Running time = %d" % (running_time * 1000), "ms")


if __name__ == "__main__":
    main()
