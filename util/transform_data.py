import numpy as np
import os
import h5py


def read_file(dataFile, fileName):
    """
    Read the data from the file path
    :param dataFile: the file path
    :param fileName: the name of the file
    :return: the data
    """
    with h5py.File(dataFile, 'r') as f:
        f.keys()
    data = h5py.File(dataFile, 'r')[fileName][()]
    return np.array(data)

def get_dataset(filePath):
    """
    Get the dataset from the file path
    :param filePath: the file path
    :return: the dataset
    """

    dataFile = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    labelFile = 'y_stim'
    data = read_file(filePath + 'd1' + '.mat', 'd1')
    for i in range(1, len(dataFile)):
        d = read_file(filePath + dataFile[i] + '.mat', dataFile[i])
        data = np.concatenate((data, d), axis=2)
    label = read_file(filePath + labelFile + '.mat', labelFile)
    x = data.transpose((2, 1, 0))
    label = label.transpose((1, 0))
    y = label[:, 1:]
    child_label = label[:, 0]
    print(type(x))
    return x, y, child_label

def process_dataset(X, Y, label):
    """
    Process the dataset
    :param X: the dataset
    :param Y: the labels
    :param label: the child labels
    :return: the processed dataset
    """

    x = []
    y_b = []
    yt = []
    yid = []
    hc = []

    for i in range(33902):
        if np.argmax(Y[i]) == 0:
            if label[i] not in yid:
                if len(hc) != 0:
                    x.append(np.array(hc))
                yid.append(np.array(label[i]))
                y_b.append(0)
                yt.append(0)
                hc = []
            hc.append(X[i])

        if np.argmax(Y[i]) == 1:
            if label[i] + 44 not in yid:
                if len(hc) != 0:
                    x.append(np.array(hc))
                yid.append(label[i] + 44)
                y_b.append(1)
                yt.append(1)
                hc = []
            hc.append(X[i])

        if np.argmax(Y[i]) == 2:
            if label[i] + 96 not in yid:
                if len(hc) != 0:
                    x.append(np.array(hc))
                yid.append(label[i] + 96)
                y_b.append(1)
                yt.append(2)
                hc = []
            hc.append(X[i])

    x.append(np.array(hc))

    x = np.array(x, dtype=object)
    y_b = np.array(y_b)
    yt = np.array(yt)
    yid = np.array(yid)
    print(np.shape(x), np.shape(y_b), np.shape(yt), np.shape(yid))
    return x, y_b, yt, yid

def _transform_data(root_dir):
    read_dir = os.path.join(root_dir, 'origin')
    write_dir = os.path.join(root_dir, 'dataset')
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    x, y, child_label = get_dataset(read_dir)
    dataset, binary_label, Triple_label, id_label = process_dataset(x, y, child_label)
    np.save("binary_label.npy", binary_label.astype(np.float32))
    np.save("triple_label.npy", Triple_label.astype(np.float32))
    np.save("id_label.npy", id_label.astype(np.float32))

    