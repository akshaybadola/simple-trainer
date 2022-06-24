from typing import Union, List, Optional, Any, Callable, Iterable, Dict, Tuple
import torch
from .models import UpdateFunction


def correct_topk(outputs: torch.FloatTensor,
                 labels: torch.LongTensor, k: int = 1) -> int:
    """Total number of correct output values along the batch dimension.

    An instance is correct if `(argmax(output[i]) in labels[i][:k]) == True` for `i`.

    `label[i]` can have multiple labels for a single data point
    and the model has to predict only one of them.

    So if there are 5 labels for each instance `argmax(output[i]) in labels[i]`
    checks for all 5. While if `k == 3`, `argmax(output[i]) in labels[i][:3]` is
    checked.

    It is assumed that the labels are ordered from most probable to least.

    Args:
        outputs: outputs from a model with shape BxO
        labels: column vector of shape BxL
        k: int <= L

    The values are matched by taking an `torch.argmax` along the other axis than
    the batch.

    """
    predictions = outputs.argmax(1)
    return sum([(p in l[:k]) for p, l in zip(predictions, labels)])


# TODO: Not sure what this does exactly
def topk_correct(outputs: torch.FloatTensor, labels: torch.LongTensor,
                 k: int = 1) -> int:
    _, indices = torch.topk(outputs, k, 1)
    equal = labels.repeat(k).reshape(-1, indices.shape[0]).T == indices
    return int(torch.sum(equal).item())


def correct(outputs: torch.FloatTensor,
            labels: torch.LongTensor) -> int:
    """Return number of correct output values along the batch dimension.

    An instance is correct if `argmax(output[i]) == label[i]` for `i`.

    Args:
        outputs: outputs from a model with shape BxO
        labels: column vector of shape Bx1

    """
    return int(torch.sum(outputs.argmax(1) == labels).item())


# TODO: Not sure if this is correct
def prec(output: torch.FloatTensor,
         labels: torch.LongTensor,
         topk: Union[Tuple, List] = (1,)) -> List[float]:
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# TODO: Again, not sure if this is correct
def accuracy(output: torch.FloatTensor,
             labels: torch.LongTensor) -> float:
    return prec(output, labels, (1,))[0]


class SimpleUpdateFunction(UpdateFunction):
    def __init__(self):
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, x: bool):
        self._train = x

    @property
    def returns(self):
        return self._returns


class ClassificationFunc(UpdateFunction):
    def __init__(self, use_correct=False):
        self._train = True
        if use_correct:
            self._returns = ["loss", "correct", "total"]
        else:
            self._returns = ["loss", "accuracy", "total"]

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, x: bool):
        self._train = x

    @property
    def returns(self):
        return self._returns

    def __call__(self, batch: Tuple[torch.FloatTensor, torch.LongTensor],
                 criterion: Union[torch.nn.Module, Callable],
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,  # type: ignore
                 **kwargs):
        inputs, labels = batch
        inputs, labels = model.to_(inputs), model.to_(labels)
        if self.train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if self.train:
            loss.backward()
            optimizer.step()
        total_correct = correct(outputs, labels)
        if self.use_correct:
            return {"loss": loss.item(),
                    "correct": total_correct,
                    "total": inputs.size()[0]}
        else:
            return {"loss": loss.item(),
                    "accuracy": float(total_correct/inputs.size()[0]),
                    "total": inputs.size()[0]}
