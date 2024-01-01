import os
import argparse
import torch
import time
import numpy as np
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from Model import Classifier

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
            tensor = torch.from_numpy(eeg_data).unsqueeze(0).unsqueeze(0)
            train_data.append(tensor)
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
    test_data_path = os.path.join(root_dir, data_path)
    test_label_path = os.path.join(root_dir, label_path)

    label_id = np.load(test_label_path)


    test_data = []
    label = []

    for children in tqdm(os.listdir(test_data_path), desc= 'Processing', unit= 'child'):
        folder_path = os.path.join(test_data_path, children)
        if int(children) in test_id:
            child_data = load_data(folder_path)
            test_data.extend(child_data)
            for i in range(len(child_data)):
                label_np = np.zeros(num_classes)
                label_np[label_id[int(children)]] = 1
                label.append(label_np)
    test_data = torch.stack(test_data)
    # test_data = test_data.unsqueeze(1)
    label = np.array(label)

    
    # Load model
    model = Classifier(None, None, None, None)
    assert args.ckpt is not None, 'checkpoint file does not exist'
    weight_path = args.ckpt
    new_weights = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(new_weights)
    model.eval()
    model.cuda(2)

    correct_predictions = 0

    for i in tqdm(range(len(test_data))):
        output = F.softmax(model(test_data[i].cuda(2)), dim=1)
        predicted_class = np.argmax(output.detach().cpu().numpy().reshape(3,), axis=-1)
        true_class = np.argmax(label[i], axis=-1)

        if predicted_class == true_class:
            correct_predictions += 1

    overall_accuracy = correct_predictions / len(test_data)

    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    args = parser.parse_args()
    main(args)