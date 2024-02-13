import torch
import torch.nn as nn


class PlantCNN(nn.Module):
    def __init__(self):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=32, stride=8)
        self.pool1 = nn.AvgPool2d(kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=15, stride=8)
        self.flatter = nn.Flatten()
        self.fc1 = nn.Linear(40, 2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.pool1(x)
        x = self.flatter(x)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    random_tensor = torch.rand((10,1,125,75), dtype=torch.float32)
    print(random_tensor.shape)
    model = PlantCNN()
    out = model(random_tensor)
    print(out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of learnable parameters: {total_params}")