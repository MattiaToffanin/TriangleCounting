from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand


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
    edges = rawData.flatMap(lambda x: (int(x.split(',')[0]), int(x.split(',')[1]))).cache()
    edges = edges.repartition(C)

    # Print parameters info
    print("Dataset = ", data_path)
    print("Number of Edges = ", edges.count())
    print("Number of Colors = ", C)
    print("Number of Repetitions = ", R)


if __name__ == "__main__":
    main()
