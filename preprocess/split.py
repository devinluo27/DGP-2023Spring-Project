import numpy as np
import os

if __name__=="__main__":
    path = os.path.join("..", "..", "ShapeNetDGP")
    paths = os.listdir(path)
    #print(paths)
    train = 100
    val = 103
    test = 138
    with open(os.path.join(".", 'train.txt'), 'w') as f:
        for i in range(0, train, 1):
            f.write(f"{paths[i]}\n")
    with open(os.path.join(".", 'val.txt'), 'w') as f:
        for i in range(train, val, 1):
            f.write(f"{paths[i]}\n")
    with open(os.path.join(".", 'test.txt'), 'w') as f:
        for i in range(val, test, 1):
            f.write(f"{paths[i]}\n")
