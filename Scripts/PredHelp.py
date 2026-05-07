import torch
from Scripts import Training
import numpy as np

model = Training.Model()
model.load_state_dict(torch.load("../TrainedModels/model1.pth"))
model.eval()

with torch.no_grad():
    file = np.load(f"{"../Dataset/Казвам се"}/s{1}.npy")
    X = torch.tensor(file, dtype=torch.float32).unsqueeze(0)

    out = model(X)
    pred = torch.argmax(out, dim=1)
    print(out)