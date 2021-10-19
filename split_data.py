import pandas as pd
import random
import numpy as np
from tqdm import tqdm

file = pd.read_csv("data_labels.csv", delimiter=";")

def random_choose_label(file):
    for index, row in tqdm(file.iterrows()):
        row[0] = row[0] + '.avi'
        if len(list(row[1])) != 1:
            label = row[1]
            label = list(label.replace(":", ""))
            row[1] = random.choice(label)

    return file


def split_data(file):
    train, val, test = np.split(file.sample(frac=1), [int(.9*len(file)), int(.95*len(file))])
    print(train.shape, val.shape, test.shape)
    return train, val, test


def save(train, val, test):
    train.to_csv('train.csv', sep=';')
    val.to_csv('val.csv', sep=';')
    test.to_csv('test.csv', sep=';')


file = random_choose_label(file)
train, val, test = split_data(file)
save(train, val, test)
