import torch
import torch.nn as nn


class PlantANN(nn.Module):
    def __init__(self):
        super(PlantANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(51,10),
            nn.LeakyReLU(),
            nn.Linear(10,2)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    random_tensor = torch.rand((100,52), dtype=torch.float32)
    model = PlantANN()
    out = model(random_tensor)
    print(out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {total_params}")