# distributed-tsp
First you should run dask scheduler by "dask scheduler --host ip:port" and then it's necessary to run at least one dask worker and assign the worker(s) to dask scheduler by "dask worker (ip:port of scheduler)".


This program solves TSPs distributedly by applying k-means to initial cities, then assigning clusters to dask workers. Every dask worker computes tsp tour for assigned cluster by using google or-tools. After all, clusters are connected by the approach discribed in the "Clustering Evolutionary Computation for Solving Travelling Salesman Problems" paper.
