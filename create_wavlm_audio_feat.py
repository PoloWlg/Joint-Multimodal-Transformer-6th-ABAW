import os
import pandas as pd
import torch
import numpy as np


def write_wavlm_feat():
    # Source folder containing the files
    source_folder = '/home/ens/AS08960/datasets/Affwild_Audio/extracted'

    # Destination folder where the new folders will be created
    destination_folder = '/home/ens/AS08960/datasets/Affwild_Audio/wavlm_feat'

    # Iterate over files in the source folder
    for feat_filename in os.listdir(source_folder):
        feat_path = os.path.join(source_folder, feat_filename)
        if os.path.isfile(feat_path):
            folder_name = os.path.splitext(feat_filename)[
                0]  # Get file name without extension
            folder_path = os.path.join(destination_folder, folder_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            all_feat = pd.read_csv(feat_path)
            all_tensor_feat = torch.tensor(all_feat.values)
            all_tensor_feat = all_tensor_feat[:, :512]

            # feat_counter = 1
            for feat_counter in range(all_tensor_feat.size(0)):
                single_audio = all_tensor_feat[feat_counter, :512]
                filee = folder_path + '/' + (str(feat_counter + 1)) + '.npy'
                np.save(filee, single_audio.numpy())


def count_files(source_folder):
    train_set = '/home/ens/AS08960/datasets/Affwild_Audio/Train_Set'
    counter = 0
    for filename in os.listdir(train_set):
        video_name = os.path.splitext(filename)[0]
        feat_path = os.path.join(source_folder, video_name)
        if not os.path.exists(feat_path + '.wavlmpooled'):
            print(video_name)
            csv_path = os.path.join(train_set, filename)
            # os.remove(csv_path)
            counter = counter + 1
    print(counter)