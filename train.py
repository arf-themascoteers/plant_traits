from plant_dataset import PlantDataset
from torch.utils.data import DataLoader
from plant_ann import PlantANN
import torch
import torch.nn as nn
import torch.optim as optim


def train():
    model = PlantANN()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ds = PlantDataset(is_train=True)
    dl = DataLoader(ds, batch_size=100, shuffle=True)
    loss = 0
    for epoch in range(100):
        for inputs, labels in dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item()}")
        print(f"After {epoch+1} epoch, loss: {loss.item()}")
    torch.save(model.state_dict(), 'saved.pth')


if __name__ == "__main__":
    train()