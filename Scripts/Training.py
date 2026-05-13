import torch
from torch import nn
import LoadDataset
import numpy as np


class Model (nn.Module):
    def __init__(self, input_size=225, hidden_size=512, num_layers=2, num_classes=3):
        super().__init__()

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, X):
        out, (hs, cs) = self.LSTM(X)

        x = hs[-1]
        out = self.fc(x)
        return out

if __name__ == "__main__":

    model = Model()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data_loader, test_data_loader = LoadDataset.CreateTrainingDataset(3, 59)

    for epoch in range(30):
        for batch in train_data_loader:
            X = batch[0]
            y = batch[1]
            output = model(X)

            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(loss)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_data_loader:
            output = model(X)
            preds = torch.argmax(output, axis=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    torch.save(model.state_dict(), "../TrainedModels/model6.pth")

    print(correct/total)
