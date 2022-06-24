from typing import Union, List, Optional, Any, Callable, Iterable, Dict

import os
from functools import partial
from pathlib import Path
import inspect

import torch

from common_pyutil.decorators import Tag
from common_pyutil.monitor import Timer
from common_pyutil.log import get_file_and_stream_logger

from .pipeline import Hooks

from .models import TrainerParams, DDPParams, UpdateFunction
from .functions import (pre_train_log, save_checkpoint,
                        pre_eval_log, post_epoch_log,
                        pre_batch_init_batch_vars,
                        post_batch_update_batch_vars,
                        post_epoch_reset_batch_vars,
                        maybe_test, maybe_validate,
                        initialize_seed, dump_state,
                        update_metrics, save_best)


def increment_epoch(self):
    self._epoch += 1


cmd = Tag("cmd")


class Trainer(Hooks):
    """:class:`Trainer` is a kind of :class:`Hooks` with certain functions.

    It has different initialization and intermediate functions specific to
    training a `Deep Learning` model.

    """
    def __init__(self, name, trainer_params: Dict[str, Any],
                 optimizer, model, data: Dict[str, Any], dataloaders,
                 update_function: Dict[str, UpdateFunction],
                 criterion, savedir: Union[str, Path] = "saves",
                 logdir: Union[str, Path] = "logs",
                 ddp_params: Optional[Dict[str, Any]] = {},
                 extra_opts: Dict[str, Any] = {},
                 post_init_hooks: List[Callable] = []):
        self._logdir = Path(logdir)
        if not self._logdir.exists():
            os.mkdir(self._logdir)
        self.log_file, self.logger = get_file_and_stream_logger(str(self._logdir), name,
                                                                name, "debug",
                                                                "info", "debug",
                                                                new_file=True)
        self._name = name
        self.data = data
        self._dataloaders = dataloaders
        self.update_function = update_function
        self._model = model
        self._trainer_params = TrainerParams(**trainer_params)
        self._criterion = criterion
        self._optimizer = optimizer
        if ddp_params:
            self._ddp_params: Optional[DDPParams] = DDPParams(**ddp_params)
        else:
            self._ddp_params = None
        self._epoch = 0
        self._init_model()
        self._init_metrics()
        self._init_hooks()
        self.timer = Timer()
        self._savedir = Path(savedir)
        if not self.savedir.exists():
            os.mkdir(self.savedir)
        self._checkpoint_prefix = "checkpoint"
        self._save_best_prefix = "best_save"
        if self.trainer_params.resume_checkpoint:
            self._resume_path: Optional[Path] = self.savedir.joinpath(
                self.trainer_params.resume_checkpoint)
        elif self.trainer_params.resume_best:
            if not bool([*self.savedir.iterdir()]):
                msg = f"No saves in savedir yet"
                self.logger.debug(msg)
            else:
                save_files = [*self.savedir.iterdir()]
                best_saves = [save for save in save_files
                              if save.name.startswith(self._save_best_prefix)]
                if best_saves:
                    self._resume_path = self.savedir.joinpath(best_saves[0])
                else:
                    self._resume_path = None
        elif self.trainer_params.resume:
            self._resume_path = self.savedir.joinpath(self.checkpoint_name)
        else:
            self._resume_path = None
        for func in post_init_hooks:
            self.add_to_hook("post_init_hook", func, "last")
        self._batch_vars: Dict[str, Any] = {}
        self.extra_opts = extra_opts
        self.run_hook("post_init_hook")

    def _init_model(self):
        if not hasattr(self._model, "model_name"):
            raise AttributeError("Model must have attribute 'model_name'")
        model_name = self._model.model_name
        self._has_cuda = self.trainer_params.cuda and torch.cuda.is_available()
        if self._has_cuda:
            self._gpus = self.trainer_params.gpus
        else:
            self._gpus = [-1]
        if self._gpus != [-1]:
            self._model = self._model.to(torch.device(f"cuda:{self._gpus[0]}"))
            if len(self._gpus) > 1:
                self._model = torch.nn.DataParallel(self._model, self._gpus)
                self._model.model_name = model_name
            self._model.to_ = lambda x: x.to(torch.device(f"cuda:{self._gpus[0]}"))
        elif self.ddp_params and self._ddp_gpu:
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model, device_ids=[self._ddp_gpu])
            self._model.model_name = model_name
            self._model.to_ = lambda x: x.to(torch.device(f"cuda:{self._ddp_gpu}"))
        else:
            self._model.to_ = lambda x: x.to(torch.device("cpu"))

    def pp(self, func):
        return self._prepare_function(func)

    def _prepare_function(self, func):
        return partial(func, self)

    def _init_hooks(self):
        """Initialize all hooks.

        For the :class:`Trainer` each function in a hook must take an instance
        of :class:`trainer` itself as the first argument.

        """
        super().__init__(self.logger)
        self._hooks = {"post_init_hook": [self.pp(initialize_seed)],
                       "pre_resume_hook": [],
                       "post_resume_hook": [],
                       "pre_save_hook": [],
                       "post_save_hook": [],
                       "pre_eval_hook": [self.pp(pre_eval_log)],
                       "post_eval_hook": [self.pp(update_metrics)],
                       "pre_batch_hook": [self.pp(pre_batch_init_batch_vars)],
                       "post_batch_hook": [self.pp(post_batch_update_batch_vars)],
                       "pre_training_hook": [self.pp(pre_train_log)],
                       "post_training_hook": [],
                       "pre_epoch_hook": [],
                       "post_epoch_hook": [*map(self.pp,
                                                [post_epoch_log,
                                                 partial(update_metrics, loop="train"),
                                                 maybe_validate, maybe_test,
                                                 increment_epoch, dump_state,
                                                 save_checkpoint, save_best,
                                                 post_epoch_reset_batch_vars])]}
        # CHECK: Do we really need to do this?
        for hook in self._hooks:
            setattr(self, f"run_{hook}", partial(self.run_hook, hook))
        cmd.add(self.add_to_hook)
        cmd.add(self.add_to_hook_before)
        cmd.add(self.add_to_hook_after)
        cmd.add(self.remove_from_hook)
        cmd.add(self.remove_from_hook_at)
        cmd.add(self.describe_hook)

    @property
    def name(self) -> str:
        return self._name

    @property
    def checkpoint_name(self) -> str:
        return "_".join([self._checkpoint_prefix, self._name,
                         self._model.model_name, self.data["name"]]).replace(" ", "_")  # type: ignore

    @property
    def save_best_name(self) -> str:
        return "_".join([self._save_best_prefix, self._name,
                         self._model.model_name, self.data["name"]]).replace(" ", "_")  # type: ignore

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def data(self) -> Dict[str, Union[str, torch.utils.data.Dataset]]:
        return self._data

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
    def dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        return self._dataloaders

    @property
    def ddp_params(self) -> Optional[DDPParams]:
        return self._ddp_params

    @ddp_params.setter
    def ddp_params(self, x):
        self._ddp_params = DDPParams(**x)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def update_function(self):
        return self._update_function

    # TODO: Either we require that it implement UpdateFunction or that it
    #       conforms to the signature
    @update_function.setter
    def update_function(self, func):
        if not isinstance(func, UpdateFunction):
            self.logger.error(f"{func} must be an instance of {UpdateFunction}")
            # FIXME: This doesn't do anything right now
        if not callable(func):
            self.logger.error(f"Not Callable {func}")
        elif not hasattr(func, "train"):
            self.logger.error(f"Attribute train not in {func}")
        elif not hasattr(func, "returns"):
            self.logger.error(f"Attribute returns not in {func}")
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
    def savedir(self) -> Path:
        """Return the directory where all the dumps and checkpoints are stored.
        """
        return self._savedir

    @savedir.setter
    def savedir(self, x: Union[str, Path]):
        self._savedir = Path(x)
        if not self._savedir.exists():
            os.mkdir(self._savedir)

    @property
    def logdir(self) -> Path:
        """Return the directory where the logfiles are stored.

        There's no setter for this property as logger is initialized early.
        """
        return self._logdir

    @property
    def criterion(self):
        """Return the current criterion.

        Criterion can be changed during training but that is not advisable and
        should be done with caution.

        """
        return self._criterion

    @property
    def trainer_params(self) -> TrainerParams:
        """Return all the :class:`Trainer`'s training specific paramters.


        """
        return self._trainer_params

    @trainer_params.setter
    def trainer_params(self, params):
        self._trainer_params = TrainerParams(**params)

    @property
    def metrics(self) -> Dict[str, Dict[str, Dict]]:
        return self._metrics

    @property
    def batch_vars(self) -> Dict[str, Dict]:
        return self._batch_vars

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def resume_path(self) -> Optional[Path]:
        return self._resume_path

    def _init_metrics(self):
        self._metrics: Dict[str, Dict[str, Dict]] = {}
        for x in ["train", "val", "test"]:
            self._metrics[x] = {m: {} for m in self.trainer_params.metrics}
            self._metrics[x]["time"] = {}

    def try_resume(self):
        self.run_hook("pre_resume_hook")
        if not self.resume_path:
            self.logger.debug("Resume not given. Not resuming")
            return
        resume_path = self.resume_path
        self.logger.debug(f"Trying to resume from {resume_path}")
        if not resume_path.suffix == ".pth":
            resume_path = resume_path.with_suffix(".pth")
        if not resume_path.exists():
            self.logger.info(f"Resume path {resume_path} doesn't exist")
            return
        self.logger.info(f"Resuming from {resume_path}")
        saved_state = torch.load(str(resume_path), map_location="cpu")
        for x in ['model_state_dict', 'optimizer_state_dict', 'metrics', 'data_name',
                  'epoch', 'model_name', 'params', 'extra_opts']:
            if x not in saved_state:
                raise AttributeError(f"Key {x} not in saved state")
        if saved_state['model_name'] != self.model.model_name:
            raise AttributeError("Error. Trying to load from a different model")
        if saved_state["data_name"] != self.data["name"]:
            raise AttributeError("Error. Trying to load a different dataset")
        self._epoch = saved_state["epoch"]
        self.logger.info(f"Resuming from checkpoint at _epoch {self.epoch}")
        if isinstance(self.model, torch.nn.parallel.DataParallel):
            self.model.module.load_state_dict(saved_state['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state['model_state_dict'])
        self.optimizer.load_state_dict(saved_state['optimizer_state_dict'])
        self._metrics = saved_state["metrics"]
        self.trainer_params = saved_state["params"]
        self.extra_opts = saved_state["extra_opts"]
        self.run_hook("post_resume_hook")

    def _save(self, name):
        self.run_hook("pre_save_hook")
        save_name = name if name.endswith(".pth") else name + ".pth"
        if isinstance(self.model, torch.nn.parallel.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save({'model_state_dict': state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': self.metrics,
                    "epoch": self.epoch,
                    'data_name': self.data["name"],
                    'model_name': self._model.model_name,
                    'params': self.trainer_params.__dict__,
                    'extra_opts': self.extra_opts},
                   self.savedir.joinpath(save_name))
        self.run_hook("post_save_hook")

    def _eval(self, val_test, debug=False):
        loader = self.dataloaders[val_test]
        self.run_hook_with_args("pre_eval_hook", loop=val_test)
        total_iters = len(loader)
        it = loader.__iter__()
        i = 0
        try:
            while True:
                with self.timer:
                    batch = it.__next__()
                self.eval_one_batch(val_test, i, total_iters, batch_time=self.timer.time,
                                    batch=batch)
                i += 1
        except StopIteration:
            pass
        # for i, batch in enumerate(loader):
        #     self.eval_one_batch(val_test, i, batch)
        self.run_hook_with_args("post_eval_hook", loop=val_test)

    def validate(self):
        self._eval("val")

    def test(self):
        self._eval("test")

    def eval_one_batch(self, val_or_test, batch_num, total_iters, batch_time, batch):
        self.run_hook_with_args("pre_batch_hook", loop=val_or_test)
        with torch.no_grad():
            with self.timer:
                self.update_function.train = False
                self.model.eval()
                with torch.no_grad():
                    retval = self.update_function(batch=batch, criterion=self.criterion,
                                                  model=self.model, optimizer=self.optimizer,
                                                  trainer=self)
        retval.update(self.timer.as_dict)
        self.run_hook_with_args("post_batch_hook", loop=val_or_test, retval=retval,
                                batch_num=batch_num, total_iters=total_iters,
                                batch_time=batch_time)

    def train_one_batch(self, batch_num, total_iters, batch_time, batch):
        self.run_hook_with_args("pre_batch_hook", loop="train")
        with self.timer:
            self.update_function.train = True
            self.model.train()
            retval = self.update_function(batch=batch, criterion=self.criterion,
                                          model=self.model, optimizer=self.optimizer,
                                          trainer=self)
        retval.update(self.timer.as_dict)
        self.run_hook_with_args("post_batch_hook", loop="train", retval=retval,
                                batch_num=batch_num, total_iters=total_iters,
                                batch_time=batch_time)

    def run_one_epoch(self):
        self.logger.info(f"Training epoch {self.epoch+1}")
        self.run_hook("pre_epoch_hook")
        total_iters = len(self.dataloaders["train"])
        it = self.dataloaders["train"].__iter__()
        i = 0
        try:
            while True:
                with self.timer:
                    batch = it.__next__()
                self.train_one_batch(i, total_iters, batch_time=self.timer.time,
                                     batch=batch)
                i += 1
        except StopIteration:
            pass
        # for i, batch in enumerate(self.dataloaders["train"]):
        #     self.train_one_batch(i, batch)
        self.run_hook("post_epoch_hook")

    def train(self):
        self.run_hook("pre_training_hook")
        while self.epoch < self.trainer_params.max_epochs:
            self.run_one_epoch()
        self.run_hook("post_training_hook")
        self.logger.info('Finished training')

    def start(self):
        self.try_resume()
        self.train()
