import numpy as np


def print_size_stats(size_list: list):
    avg = np.average(size_list)
    min = np.min(size_list)
    max = np.max(size_list)
    mid = np.median(size_list)
    counts = np.bincount(size_list)
    mode = np.argmax(counts)
    mode_count = counts[2]
    print('avg: {:3f}, min: {:d}, max: {:d}, mid: {:d}, mode: {:d}, mode_count: {:d}'
          .format(avg, int(min), int(max), int(mid), int(mode), int(mode_count)))


def print_stats_rate(size_list: list):
    counts = np.bincount(size_list)
    t1 = np.argsort(counts)[-1]
    t2 = np.argsort(counts)[-2]
    t3 = np.argsort(counts)[-3]
    r1 = float(counts[t1] / len(size_list))
    r2 = float(counts[t2] / len(size_list))
    r3 = float(counts[t3] / len(size_list))
    print('top 1 stat: {:d}, top 1 rate: {:3f}'.format(t1, r1))
    print('top 2 stat: {:d}, top 2 rate: {:3f}'.format(t2, r2))
    print('top 3 stat: {:d}, top 3 rate: {:3f}'.format(t3, r3))


