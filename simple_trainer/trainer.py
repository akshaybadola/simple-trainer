from typing import Union, List, Optional, Any, Callable, Iterable, Dict
import os
import torch
import inspect
from functools import partial

from flask import Flask, request, Response

from .helpers import gen_file_and_stream_logger
from .models import TrainerParams
from .hooks import (pre_train_log, save_checkpoint,
                    pre_eval_log, post_epoch_log,
                    pre_batch_init_batch_vars,
                    post_batch_update_batch_vars,
                    post_epoch_reset_batch_vars,
                    maybe_test, maybe_validate,
                    initialize_seed,
                    update_metrics, save_best)


def increment_epoch(self):
    self.epoch += 1


class Trainer:
    def __init__(self, name, trainer_params, optimizer, model, data,
                 dataloaders, update_function, criterion, extra_opts):
        self.log_file, self.logger = gen_file_and_stream_logger("logs", name,
                                                                name, "debug",
                                                                "info", "debug")
        self._name = name
        self.data = data
        self._dataloaders = dataloaders
        self.update_function = update_function
        self._model = model
        self.trainer_params = trainer_params
        self._criterion = criterion
        self._optimizer = optimizer
        self.epoch = 0
        self._init_model()
        self._init_metrics()
        self._init_hooks()
        self._savedir = "saves"
        if not os.path.exists(self._savedir):
            os.mkdir(self._savedir)
        self._checkpoint_name = "checkpoint"
        self._save_best_name = "best_save"
        if self.trainer_params.resume_checkpoint:
            self._resume_path = os.path.join(self._savedir, self.trainer_params.resume_checkpoint)
        elif self.trainer_params.resume_best:
            if not (os.path.exists(os.path.join(self._savedir)) and os.listdir(self._savedir)):
                msg = f"No saves in savedir yet"
                self.logger.debug(msg)
            else:
                save_files = os.listdir(self._savedir)
                best_saves = [save for save in save_files if save.startswith(self._save_best_name)]
                if best_saves:
                    self._resume_path = os.path.join(self._savedir, best_saves[0])
                else:
                    self._resume_path = None
        else:
            self._resume_path = None
        self.run_hook("post_init_hook")
        self.extra_opts = extra_opts
        self._try_resume()

    def _init_model(self):
        self._has_cuda = self.trainer_params.cuda and torch.cuda.is_available()
        if self._has_cuda:
            self._gpus = self.trainer_params.gpus
        else:
            self._gpus = [-1]
        if self._gpus != [-1]:
            self._model = self._model.to(torch.device(f"cuda:{self._gpus[0]}"))
            if len(self._gpus) > 1:
                self._model = torch.nn.DataParallel(self._model, self._gpus)
            self._model.to_ = lambda x: x.to(torch.device(f"cuda:{self._gpus[0]}"))
        else:
            self._model.to_ = lambda x: x.to(torch.device("cpu"))

    def _init_hooks(self):
        self._hooks = {"post_init_hook": [initialize_seed],
                       "pre_resume_hook": [],
                       "post_resume_hook": [],
                       "pre_save_hook": [],
                       "post_save_hook": [],
                       "pre_eval_hook": [pre_eval_log],
                       "post_eval_hook": [update_metrics],
                       "pre_batch_hook": [pre_batch_init_batch_vars],
                       "post_batch_hook": [post_batch_update_batch_vars],
                       "pre_training_hook": [pre_train_log],
                       "post_training_hook": [update_metrics],
                       "pre_epoch_hook": [],
                       "post_epoch_hook": [post_epoch_log,
                                           partial(update_metrics, loop="train"),
                                           maybe_validate, maybe_test,
                                           save_checkpoint, save_best,
                                           increment_epoch,
                                           post_epoch_reset_batch_vars]}
        for hook in self._hooks:
            setattr(self, f"run_{hook}", partial(self.run_hook, hook))

    def run_hook(self, hook):
        hook = self._hooks.get(hook, None)
        if hook:
            for func in hook:
                func(self)

    def run_hook_with_args(self, hook, *args, **kwargs):
        hook = self._hooks.get(hook, None)
        if hook:
            for func in hook:
                func(self, *args, **kwargs)

    def add_hook(self, hook, func, position: Union[int, str] = 0):
        """Add function :code:`func` to hook with name `hook`.

        Args:
            hook: Name of the hook
            func: A function with a single argument, which is the trainer instance
            position: Where to insert the hook. Defaults to front of list.
        """
        if hook in self._hooks:
            self.logger.info(f"Adding function {func.__name__} to {hook} at {position}")
            if position == "first":
                pos = 0
            elif position == "last":
                pos = len(hook)
            else:
                pos = position
            self._hooks[hook].insert(pos, func)

    def remove_hook(self, hook, function_name):
        hook = self._hooks.get(hook, None)
        if hook:
            self._hooks[hook] = [*filter(lambda x: x.__name__ == function_name, hook)]

    def remove_hook_at(self, hook, position):
        hook = self._hooks.get(hook, None)
        if hook:
            hook.pop(position)

    @property
    def hooks(self):
        return self._hooks

    def describe_hook(self, hook):
        hook = self._hooks.get(hook, None)
        if hook:
            return [f"partial({x.func.__name__}, {x.args}, {x.keywords})"
                    if isinstance(x, partial) else x.__name__
                    for x in hook]
        else:
            return None

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def dataloaders(self):
        return self._dataloaders

    @data.setter
    def data(self, data):
        if "name" not in data:
            self.logger.error("Dataset must have a name")
            return
        if data["train"] is None:
            self.logger.error("Training data cannot be None")
            return
        for x in ["train", "val", "test"]:
            if data[x] is not None:
                if not iter(data[x]):
                    self.logger.error("Dataset must be iterable")
                else:
                    try:
                        len(data[x])
                    except TypeError:
                        self.logger.error("Dataset must have length")
                        return
        self._data = data

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def update_function(self):
        return self._update_function

    @update_function.setter
    def update_function(self, func):
        if not callable(func):
            self.logger.error(f"Not Callable {func}")
        elif not hasattr(func, "train"):
            self.logger.error(f"Attribute train not in {func}")
        else:
            sig = inspect.signature(func).parameters
            if "batch" not in sig:
                self.logger.error(f"{func} does not take batch as input")
            elif "model" not in sig:
                self.logger.error(f"{func} does not take model as input")
            elif "optimizer" not in sig:
                self.logger.error(f"{func} does not take optimizer as input")
            elif "criterion" not in sig:
                self.logger.error(f"{func} does not take criterion as input")
            else:
                self.logger.info(f"Setting update_function to {func}")
                self._update_function = func

    @property
    def criterion(self):
        return self._criterion

    @property
    def trainer_params(self):
        return self._trainer_params

    @trainer_params.setter
    def trainer_params(self, params):
        self._trainer_params = TrainerParams(**params)

    @property
    def metrics(self):
        return self._metrics

    def _init_metrics(self):
        self._metrics = {x: {m: [] for m in self.trainer_params.metrics}
                         for x in ["train", "val", "test"]}

    def _try_resume(self):
        self.run_hook("pre_resume_hook")
        if not self._resume_path:
            self.logger.debug("Not resuming")
            return
        self.logger.info(f"Resuming from {self._resume_path}")
        saved_state = torch.load(self._resume_path, map_location="cpu")
        for x in ['model_state_dict', 'optimizer_state_dict', 'metrics', 'data_name',
                  'model_name', 'params']:
            if x not in saved_state:
                raise AttributeError(f"Key {x} not in saved state")
        if saved_state['model'] != self.model.__class__.__name__:
            raise AttributeError("Error. Trying to load from a different model")
        if saved_state["data_name"] != self.data.name:
            raise AttributeError("Error. Trying to load a different dataset")
        self.model.load_state_dict(saved_state['model_state_dict'], map_location="cpu")
        self.optimizer.load_state_dict(saved_state['optimizer_state_dict'])
        self._metrics = saved_state["metrics"]
        self.trainer_params = saved_state["params"]
        self.run_hook("post_resume_hook")

    def _save(self, name):
        self.run_hook("pre_save_hook")
        save_name = name if name.endswith(".pth") else name + ".pth"
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': self.metrics,
                    'data_name': self.data["name"],
                    'model_name': self.model.__class__.__name__,
                    'params': self.trainer_params.__dict__},
                   os.path.join(self._savedir, save_name))
        self.run_hook("post_save_hook")

    def _eval(self, val_test, debug=False):
        loader = self.dataloaders[val_test]
        self.run_hook_with_args("pre_eval_hook", val_test)
        for i, batch in enumerate(loader):
            self.eval_one_batch(val_test, i, batch)
        self.run_hook_with_args("post_eval_hook", val_test)

    def validate(self):
        self._eval("val")

    def test(self):
        self._eval("test")

    def eval_one_batch(self, val_or_test, i, batch):
        self.run_hook_with_args("pre_batch_hook", val_or_test)
        self.update_function.train = False
        retval = self.update_function(batch=batch, criterion=self.criterion,
                                      model=self.model, optimizer=self.optimizer,
                                      trainer=self)
        self.run_hook_with_args("post_batch_hook", val_or_test, retval)

    def train_one_batch(self, i, batch):
        self.run_hook_with_args("pre_batch_hook", "train")
        self.update_function.train = True
        retval = self.update_function(batch=batch, criterion=self.criterion,
                                      model=self.model, optimizer=self.optimizer,
                                      trainer=self)
        self.run_hook_with_args("post_batch_hook", "train", retval)

    def run_one_epoch(self):
        self.run_hook("pre_epoch_hook")
        for i, batch in enumerate(self.dataloaders["train"]):
            self.train_one_batch(i, batch)
        self.run_hook("post_epoch_hook")

    def train(self):
        self.run_hook("pre_training_hook")
        while self.epoch < self.trainer_params.max_epochs:
            self.run_one_epoch()
        self.run_hook("post_training_hook")
        self.logger.info('Finished training')
