import numpy as np
import os
import h5py
from tqdm import tqdm


def read_file(dataFile, fileName):
    """
    Read the data from the file path
    Args:
        dataFile: the file path
        fileName: the name of the file

    Returns: 
        the data
    """
    with h5py.File(dataFile, 'r') as f:
        f.keys()
    data = h5py.File(dataFile, 'r')[fileName][()]
    return np.array(data)


def get_dataset(filePath):
    """
    Get the dataset from the file path

    Args: 
        filePath: the file path

    Returns:  
        the dataset
    """

    dataFile = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    labelFile = 'y_stim'
    data = read_file(filePath + '/d1' + '.mat', 'd1')
    for i in tqdm(range(1, len(dataFile)), desc= 'Reading', unit= 'file'):
        d = read_file(os.path.join(filePath ,dataFile[i] + '.mat'), dataFile[i])
        data = np.concatenate((data, d), axis=2)
    label = read_file(os.path.join(filePath ,labelFile + '.mat'), labelFile)
    x = data.transpose((2, 1, 0))
    label = label.transpose((1, 0))
    y = label[:, 1:]
    child_label = label[:, 0]
    return x, y, child_label


def process_dataset(X, Y, label):
    """
    Process the dataset

    Args:
        X: the dataset
        Y: the labels
        label: the child labels

    Returns: 
        the processed dataset
    """

    x = []
    y_b = []
    yt = []
    yid = []
    hc = []

    for i in tqdm(range(33902), desc= 'Date processing', unit='trial'):
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
    return x, y_b, yt, yid


def _transform_data(root_dir):
    """
    Transforms the input data and saves the transformed data to disk.
    
    Args:
        root_dir (str): The root directory containing the input data.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input data directory is not found.

    """

    read_dir = os.path.join(root_dir, 'origin')
    write_dir = os.path.join(root_dir, 'dataset')
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    x, y, child_label = get_dataset(read_dir)
    dataset, binary_label, Triple_label, id_label = process_dataset(x, y, child_label)

    #save training data
    train_write_dir = os.path.join(write_dir, 'train')
    if not os.path.isdir(train_write_dir):
        os.makedirs(train_write_dir)

    cout = 0
    for i in tqdm(range(144), desc='Writing', unit= 'child'):
        save_path = os.path.join(train_write_dir, str(i))
        if not os.path.isdir(save_path):
                os.makedirs(save_path)
        for data in dataset[i]:
            file_path = os.path.join(save_path ,str(cout) + '.npy')
            np.save(file_path, data.astype(np.float32))
            cout+=1
        cout = 0


    # save different training label
    label_write_dir = os.path.join(write_dir,'label')
    if not os.path.isdir(label_write_dir):
        os.makedirs(label_write_dir)
    binary_label_path = os.path.join(label_write_dir, "binary_label.npy")
    Triple_label_path = os.path.join(label_write_dir, "triple_label.npy")
    id_label_path = os.path.join(label_write_dir, "id_label.npy")
    np.save(binary_label_path, binary_label.astype(np.int32))
    np.save(Triple_label_path, Triple_label.astype(np.int32))
    np.save(id_label_path, id_label.astype(np.int32))

    
    