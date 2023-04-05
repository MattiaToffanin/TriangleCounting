from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import numpy as np
import time
from collections import defaultdict


def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


def MR_ApproxTCwithNodeColors(edges, C):
    p = 8191  # Prime number
    a = rand.randint(1, p - 1)  # Random number at each invocation
    b = rand.randint(0, p - 1)  # Random number at each invocation

    def hc(u):  # Color hash function
        return ((a * u + b) % p) % C

    edges_counter = (
        edges.map(lambda v: (hc(v[0]), (v[0], v[1])) if hc(v[0]) == hc(v[1]) else (-1, 0))  # Map Phase (R1)
        .groupByKey()  # Shuffle + Grouping
        .mapValues(list).map(lambda v: (0, CountTriangles(v[1])) if v[0] != -1 else (0, 0))  # Reduce Phase (R1)
        .reduceByKey(lambda x, y: x + y))  # Reduce Phase (R2)
    return edges_counter.collect()[0][1] * C * C


def MR_ApproxTCwithSparkPartitions(edges, C):
    def list_counters(pairs):
        return [(0, CountTriangles(e[1])) for e in pairs]

    edges_counter = (edges.map(lambda e: (rand.randint(0, C - 1), e))  # Map Phase (R1)
                     .groupByKey()  # Shuffle + Grouping
                     .mapValues(list).mapPartitions(list_counters)  # Reduce Phase (R1)
                     .reduceByKey(lambda x, y: x + y))  # Reduce Phase (R2)
    return edges_counter.collect()[0][1] * C * C


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
    median_triangles = np.median(triangles_counter)
    mean_running_time = cumulative_running_time / R

    # Print Algorithm1 output
    print("Approximation through node coloring")
    print("- Number of triangles (median over", R, "runs) = %d" % median_triangles)
    print("- Running time (average over", R, "runs) = %d" % (mean_running_time * 1000), "ms")

    start_time = time.time()
    triangles_number = MR_ApproxTCwithSparkPartitions(edges, C)
    end_time = time.time()
    running_time = end_time - start_time

    # Print Algorithm2 output
    print("Approximation through Spark partitions")
    print("- Number of triangles = %d" % triangles_number)
    print("- Running time = %d" % (running_time * 1000), "ms")


if __name__ == "__main__":
    main()
