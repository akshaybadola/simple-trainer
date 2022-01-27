import torch
from simple_trainer import helpers

from util import MLP


def test_correct():
    outputs = torch.randn(32, 10)
    labels = torch.randint(10, (32,))
    assert int((outputs.argmax(1) == labels).sum()) == helpers.correct(outputs, labels)


def test_correct_topk():
    outputs = torch.randn(32, 10)
    labels = torch.randint(10, (32, 2))
    result = []
    for i, o in enumerate(outputs):
        result.append(o.argmax() in labels[i])
    assert sum(result) == helpers.correct_topk(outputs, labels, 2)


def test_classification_func():
    func = helpers.ClassificationFunc()
    inputs = torch.randn(32, 10, requires_grad=True)
    labels = torch.randint(10, (32,))
    model = MLP(10, 10)
    model.train()
    model.model_name = "MLP"
    model.to_ = lambda x: x
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    retval = func((inputs, labels), criterion, model, optimizer)
    assert all([k in func.returns for k in retval.keys()])
