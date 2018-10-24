import os
import pandas
import numpy as np


def read_all_from_directory(dir_path):
    image_files = []
    labels = []

    csv_file = [os.path.join(dir_path,x) for x in os.listdir(dir_path) if os.path.splitext(x)[1] == '.csv'][0]

    df = pandas.read_csv(csv_file)

    image_files = df['Path']

    image_files = list(map(lambda s: os.path.join(dir_path, s), image_files))

    labels = df['ClassId']

    image_files = np.array(image_files)
    labels = np.array(labels)

    return image_files, labels

