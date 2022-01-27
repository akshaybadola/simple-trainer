import pytest
import torch
import torchvision
from simple_trainer.models import TrainerParams
from simple_trainer.trainer import Trainer
from simple_trainer.helpers import ClassificationFunc
from simple_trainer.pipeline import Hooks

from util import MLP, ToyDataset, Net


@pytest.fixture
def toy_data():
    train_data = ToyDataset(100)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=10)
    val_data = ToyDataset(10)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2)
    return train_data, train_dataloader, val_data, val_dataloader


@pytest.fixture
def trainer(toy_data):
    params = {"gpus": 0, "cuda": True, "seed": 1111, "resume": False, "metrics":
              ["loss", "accuracy"], "log_frequency": 1, "test_frequency": 1,
              "val_frequency": 1, "max_epochs": 100}
    update_function = ClassificationFunc()
    from torch.optim import SGD
    model = MLP(10, 10)
    model.model_name = "MLP"
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_data, train_dataloader, val_data, val_dataloader = toy_data
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
    class Derived(Hooks):
        def _prepare_function(self, func):
            return super()._prepare_function(func)

    hooks = Derived(trainer.logger)
    return hooks
