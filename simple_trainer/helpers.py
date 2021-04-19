from typing import Union, List, Optional, Any, Callable, Iterable, Dict
import os
import sys
import logging
import torch
from contextlib import ExitStack


def call_with_contexts(func, contexts, *args, **kwargs):
    with ExitStack() as stack:
        for con in contexts:
            stack.enter_context(con)
        func(*args, **kwargs)


def correct_topk(outputs, labels, k=1):
    _, indices = torch.topk(outputs, k, 1)
    equal = labels.repeat(k).reshape(-1, indices.shape[0]).T == indices
    return torch.sum(equal).item()


def correct(outputs, labels):
    return accuracy_topk(outputs, labels)


def prec(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ClassificationFunc:
    def __init__(self):
        self.train = True
        self.returns = ["loss", "accuracy", "total"]

    def __call__(self, batch, criterion, model, optimizer, **kwargs):
        inputs, labels = batch
        inputs, labels = model.to_(inputs), model.to_(labels)
        if self.train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if self.train:
            loss.backward()
            optimizer.step()
        correct = accuracy_topk(outputs, labels)
        return {"loss": loss.item(),
                "accuracy": correct.sum().item()/len(inputs),
                "total": len(inputs)}


def rename_or_set_default(names, namespace, name, default):
    for _name in names:
        if _name in namespace:
            namespace.__dict__[name] = namespace.__dict__[_name]
            return
    namespace.__dict__[name] = default


def get_backup_num(filedir: str, filename: str) -> int:
    backup_files = [x for x in os.listdir(filedir) if x.startswith(filename)]
    backup_maybe_nums = [b.split('.')[-1] for b in backup_files]
    backup_nums = [int(x) for x in backup_maybe_nums
                   if any([_ in x for _ in list(map(str, range(10)))])]
    if backup_nums:
        cur_backup_num = max(backup_nums) + 1
    else:
        cur_backup_num = 0
    return cur_backup_num


def gen_file_and_stream_logger(logdir: str, logger_name: str,
                               log_file_name: str,
                               file_loglevel: Optional[str] = None,
                               stream_loglevel: Optional[str] = None,
                               logger_level: Optional[str] = None,
                               one_file: Optional[bool] = False):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not log_file_name.endswith('.log'):
        log_file_name += '.log'
    log_file = os.path.abspath(os.path.join(logdir, log_file_name))
    if os.path.exists(log_file) and not one_file:
        backup_num = get_backup_num(logdir, log_file_name)
        os.rename(log_file, log_file + '.' + str(backup_num))
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    if stream_loglevel is not None and hasattr(logging, stream_loglevel.upper()):
        stream_handler.setLevel(getattr(logging, stream_loglevel.upper()))
    else:
        stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    if file_loglevel is not None and hasattr(logging, file_loglevel.upper()):
        file_handler.setLevel(getattr(logging, file_loglevel.upper()))
    else:
        file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    if logger_level is not None and hasattr(logging, logger_level.upper()):
        logger.setLevel(getattr(logging, logger_level.upper()))
    else:
        logger.setLevel(logging.DEBUG)
    return log_file, logger
