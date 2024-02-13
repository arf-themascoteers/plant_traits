import torch
from plant_dataset import PlantDataset
from plant_ann import PlantANN
from torch.utils.data import DataLoader


def test():
    model = PlantANN()
    model.load_state_dict(torch.load('saved.pth'))
    model.eval()
    test_dataset = PlantDataset(is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    test()