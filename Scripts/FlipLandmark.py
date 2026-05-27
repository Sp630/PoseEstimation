import os
import numpy as np


for folder in os.listdir('../Dataset'):
    path_to_folder = os.path.join('../Dataset', folder)
    save_idx = 0
    for i in range(0, 30):
        sequence = np.load(f"{path_to_folder}/s{i + 1}.npy")
        feature = []
        for sample in sequence: #one 225 in 60
            newLandmark = sample.copy()
            newLandmark[::3] *= -1
            feature.append(newLandmark)
        arr = np.array(feature)
        new_path_to_folder = os.path.join('../LegacyDatasets/FlipDataset', folder)

        np.save(f"{new_path_to_folder}/s{save_idx}.npy", arr)
        save_idx += 1
        np.save(f"{new_path_to_folder}/s{save_idx}.npy", sequence)
        save_idx += 1
