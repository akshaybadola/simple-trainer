import torch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output


class MLP(torch.nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Linear(in_shape, out_shape), torch.nn.ReLU(False))

    def forward(self, x):
        return self.layer(x)


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_instances, shape=(10, 10)):
        self.data = (torch.randn(num_instances, *shape), torch.randint(0, 10, (num_instances,)))
        self.name = "test_data"

    def __getitem__(self, indx):
        return self.data[0][indx], self.data[1], indx

    def __len__(self):
        return self.data[0].shape[0]
