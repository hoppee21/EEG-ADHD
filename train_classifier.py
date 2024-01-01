import time
import os
import yaml
import torch
import argparse
from tqdm import tqdm
import numpy as np
from Model import Classifier
import pytorch_lightning as pl

def load_data(data_path):
    """
    Load data from data_path
    :param data_path: path to data
    :return: train_data: list of training data
    """
    train_data = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(data_path, file_name)
            eeg_data = np.load(file_path)
            train_data.append(torch.from_numpy(eeg_data))
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
    num_classes = config['EEG_net']['num_classes']
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
            for i in range(len(child_data)):
                label_np = np.zeros(num_classes)
                label_np[label_id[int(children)]] = 1
                label.append(label_np)
    train_data = torch.stack(train_data)
    train_data = train_data.unsqueeze(1)
    label = np.array(label)

    tran_model = Classifier(train_data, label, train_data, label)

    trainer = pl.Trainer(max_epochs=config['trainer']['max_epochs'],
                        accelerator=config['trainer']['accelerator'],
                        devices=[2], strategy='ddp')
    trainer.fit(tran_model)

    
    weights = tran_model.state_dict()
    torch.save(weights, config['save_ckpt_path'])

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our model')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()
    main(args)

   
        