import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch

def GetGesture(num_samples, path):
    gesture_dataset = np.zeros((num_samples, 60, 225))
    for i in range(0, num_samples):
        file = np.load(f"{path}/s{i + 1}.npy")
        gesture_dataset[i] = file
    return gesture_dataset


def LoadDataset(num_gestures, num_samples):
    dataX = []
    dataY = []
    label = 0
    for folder in os.listdir('../FlipDataset'):
        path_to_folder = os.path.join('../FlipDataset', folder)
        gesture_data = GetGesture(num_samples = num_samples, path=path_to_folder)
        dataX.append(gesture_data)
        dataY.extend([label] * num_samples)
        label += 1

    datasetX = np.array(dataX)
    datasetX = datasetX.reshape([num_gestures*num_samples, 60, 225])
    datasetY = np.array(dataY)


    return datasetX, datasetY

def CreateTrainingDataset(num_gesture, num_samples):
    X, y = LoadDataset(num_gesture, num_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle= True, random_state = 42)

    train_dataset_Loader = torch.utils.data.DataLoader(dataset=Dataset(X_train, y_train), batch_size=32, shuffle=True)
    test_dataset_Loader = torch.utils.data.DataLoader(dataset=Dataset(X_test, y_test), batch_size=32, shuffle=True)

    return train_dataset_Loader, test_dataset_Loader


class Dataset():
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = y
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.X)

if __name__ == "__main__":
    train_dataset_loader, _ = CreateTrainingDataset(2, 10)
    for batch in train_dataset_loader:
        print(batch[1])