import time
import os
import yaml
import torch
import argparse
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from Model import EEGNet_encoder
from MoCo import MoCo

def load_data(data_path):
    """
    Load data from data_path
    :param data_path: path to data
    :return: train_data: list of pairs of training data
    """
    train_data = []
    file_names = [file_name for file_name in os.listdir(data_path) if file_name.endswith('.npy')]

    for i in range(len(file_names)):
        file_path_1 = os.path.join(data_path, file_names[i])
        file_path_2 = os.path.join(data_path, file_names[(i + 1) % len(file_names)])  # Wrap around to the first file

        eeg_data_1 = np.load(file_path_1)
        eeg_data_2 = np.load(file_path_2)

         # Concatenate the two tensors along a specified dimension (e.g., dim=0 for vertical stacking)
        combined_data = torch.cat((torch.from_numpy(eeg_data_1), torch.from_numpy(eeg_data_2)), dim=0)
        train_data.append(combined_data)

    return train_data

def main(args):
    old_time = time.time()

    test_id = [18, 21, 30, 64, 71, 82, 123]

    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    # Load data
    num_classes = 3
    root_dir = config['dataset']['dataset_root_dir']
    data_path = config['dataset']['train_data_path']
    label_path = config['dataset']['train_label_path']
    train_data_path = os.path.join(root_dir, data_path)
    train_label_path = os.path.join(root_dir, label_path)

    label_id = np.load(train_label_path)

    train_data = []
    label = []

    
    for children in tqdm(os.listdir(train_data_path), desc= 'Processing', unit= 'child'):
        folder_path = os.path.join(train_data_path, children)
        if int(children) not in test_id:
            child_data = load_data(folder_path)
            train_data.extend(child_data)
            for _ in range(len(child_data)):
                label_np = np.zeros(num_classes)
                label_np[label_id[int(children)]] = 1
                label.append(label_np)
    train_data = torch.stack(train_data)
    train_data = train_data.unsqueeze(1)
    label = np.array(label)

    # Load model
    model = MoCo(EEGNet_encoder, train_data, label, mlp=True)
    trainer = pl.Trainer(max_epochs=config['trainer']['max_epochs'],
                        accelerator=config['trainer']['accelerator'],
                        devices=[2], strategy='ddp')
    
    trainer.fit(model)

    weights = model.state_dict()
    torch.save(weights, config['save_ckpt_path'])

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our model')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()
    main(args)