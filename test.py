import os
import sys
import networkx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import dataset


if __name__ == "__main__":
    data_path = "/Users/fy/Documents/CCSDATA/output/eclipse.jdt.core"
    dirs = dataset.list_atom_dirs(data_path)
    print(len(dirs))

    diffs = [dataset.MethodDiff(x) for x in dirs]
    len(diffs)