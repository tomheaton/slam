#!/usr/bin/env python
import os
import numpy as np
from new_error_func import calc_error_new
from load import load_graph_file
from plot import plot_graph
from slam import calculate_global_error, optimise_graph

# testing
# from transformation_funcs import t2v, inv_t, v2t, inv_v

def main():
    # vertices, edges = load_graph_file('../data/intel.g2o')
    # vertices, edges = load_graph_file('../data/intel_from_mat.g2o')
    vertices, edges = load_graph_file('../data/test.g2o')
    # optimise_graph(vertices, edges)
    # plot_graph(vertices, edges)
    print(edges)
    print("----------")
    print(calculate_global_error(vertices, edges))
    print(calc_error_new(vertices, edges)) # actual=10.2903

if __name__ == "__main__":
    main()

