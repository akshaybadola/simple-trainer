import pytest
import torch
import torchvision
import sys
from simple_trainer.models import TrainerParams
from simple_trainer.trainer import Trainer
from simple_trainer.helpers import ClassificationFunc
from simple_trainer.pipeline import Hooks


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
        self.layer = torch.nn.ReLU(torch.nn.Linear(in_shape, out_shape))

    def forward(self, x):
        return self.layer(x)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, num_instances):
        self.data = (torch.randn(num_instances, 10, 10), torch.randint(0, 10, (num_instances,)))
        self.name = "test_data"

    def __getitem__(self, indx):
        return self.data[0][indx], self.data[1], indx

    def __len__(self):
        return self.data[0].shape[0]


train_data = TestDataset(100)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=10)
val_data = TestDataset(10)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2)


@pytest.fixture
def trainer():
    params = {"gpus": 0, "cuda": True, "seed": 1111, "resume": False, "metrics":
              ["loss", "accuracy"], "log_frequency": 1, "test_frequency": 1,
              "val_frequency": 1, "max_epochs": 100}
    update_function = ClassificationFunc()
    from torch.optim import SGD
    model = MLP(10, 10)
    model.model_name = "MLP"
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = Trainer("test_trainer", params, optimizer, model,
                      data={"name": "test_data", "train": train_data, "val": val_data, "test": None},
                      dataloaders={"train": train_dataloader, "val": val_dataloader, "test": None},
                      update_function=update_function, criterion=torch.nn.CrossEntropyLoss())
    return trainer


@pytest.fixture
def trainer_with_mnist():
    params = {"gpus": 0, "cuda": True, "seed": 1111, "resume": False, "metrics":
              ["loss", "accuracy"], "log_frequency": 1, "test_frequency": 1,
              "val_frequency": 1, "max_epochs": 100}
    update_function = ClassificationFunc()
    from torch.optim import Adam
    model = Net()
    model.model_name = "Net"
    optimizer = Adam(model.parameters(), lr=0.01)
    data = {"name": "mnist",
            "train": torchvision.datasets.MNIST('.data',
                                                train=True,
                                                download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                ])),
            "val": None,
            "test": torchvision.datasets.MNIST('.data',
                                               train=False,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                               ]))}
    dataloaders = {"train": torch.utils.data.DataLoader(data["train"],
                                                        **{"batch_size": 32,
                                                           "num_workers": 0,
                                                           "shuffle": True,
                                                           "pin_memory": False}),
                   "val": None,
                   "test": torch.utils.data.DataLoader(data["test"],
                                                       **{"batch_size": 32,
                                                          "num_workers": 0,
                                                          "shuffle": False,
                                                          "pin_memory": False})}
    trainer = Trainer("test_trainer", params, optimizer, model,
                      data=data, dataloaders=dataloaders,
                      update_function=update_function, criterion=torch.nn.CrossEntropyLoss())
    return trainer


@pytest.fixture
def hooks(trainer):
    hooks = Hooks(trainer.logger)
    return hooks
