import os
import pandas as pd
from dep.dataset.generator import NamingIter


def save_dataset(dataset, save_dir):
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))

    dic = {}
    dic.setdefault("method_name", [])
    for i in range(48):
        dic.setdefault("walk_" + str(i), [])

    for i in range(len(dataset)):
        example = dataset.__getitem__(i)
        name = getattr(example, "method_name")
        if len(name) < 1:
            print(name)
            print("empty name, index: %d" % (i))
        dic["method_name"].append(','.join(name))
        for j in range(48):
            walk = getattr(example, "walk_" + str(j))
            dic["walk_"+str(j)].append(','.join(walk))
    df = pd.DataFrame(dic)
    df.to_csv(save_dir)
    return df

def save_dataset2(dataset, save_dir):
    print("Saving in {}...".format(save_dir))

    dic = {}
    dic.setdefault("local_name", [])
    for i in range(48):
        dic.setdefault("jimple_" + str(i), [])
        dic.setdefault("ir_" + str(i), [])
        dic.setdefault("trans_" + str(i), [])
    dic.setdefault("comment", [])

    for i in range(len(dataset)):
        example = dataset.__getitem__(i)
        name = getattr(example, "local_name")
        if len(name) < 1:
            print(name)
            print("empty name, index: %d" % (i))
        dic["local_name"].append(",".join(name))
        for j in range(48):
            jimple = getattr(example, "jimple_" + str(j))
            ir = getattr(example, "ir_" + str(j))
            trans = getattr(example, "trans_" + str(j))
            dic["jimple_"+str(j)].append(",".join(jimple))
            dic["ir_"+str(j)].append(",".join(ir))
            dic["trans_"+str(j)].append(",".join(trans))
        comment = getattr(example, "comment")
        dic["comment"].append(",".join(comment))
    df = pd.DataFrame(dic)
    df.to_csv(save_dir)

def save_dataset3(dataset, save_dir):
    print("Saving in {}...".format(save_dir))

    dic = {}
    dic.setdefault("comment", [])
    for i in range(48):
        dic.setdefault("jimple_" + str(i), [])
        dic.setdefault("ir_" + str(i), [])
        dic.setdefault("trans_" + str(i), [])

    for i in range(len(dataset)):
        example = dataset.__getitem__(i)
        name = getattr(example, "comment")
        if len(name) < 1:
            print(name)
            print("empty name, index: %d" % (i))
        dic["comment"].append(",".join(name))
        for j in range(48):
            jimple = getattr(example, "jimple_" + str(j))
            ir = getattr(example, "ir_" + str(j))
            trans = getattr(example, "trans_" + str(j))
            dic["jimple_"+str(j)].append(",".join(jimple))
            dic["ir_"+str(j)].append(",".join(ir))
            dic["trans_"+str(j)].append(",".join(trans))
    df = pd.DataFrame(dic)
    df.to_csv(save_dir)

