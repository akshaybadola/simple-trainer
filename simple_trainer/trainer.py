from typing import Union, List, Optional, Any, Callable, Iterable, Dict

import atexit
import os
from functools import partial
from pathlib import Path
import inspect

import torch

from common_pyutil.decorators import Tag
from common_pyutil.monitor import Timer
from common_pyutil.log import get_file_and_stream_logger

from .prefetch import Prefetcher
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
prop = Tag("prop")


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
                                                                new_file=False)
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
        self._maybe_init_prefetchers()
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
                msg = "No saves in savedir yet"
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
        self._extra_opts = extra_opts
        atexit.register(self._maybe_join_prefetchers)
        self.run_hook("post_init_hook")

    def _maybe_init_prefetchers(self):
        self._prefetchers = {}
        if self.trainer_params.use_prefetch:
            for k in self.dataloaders:
                if self.dataloaders[k] is not None:
                    self._prefetchers[k] = Prefetcher(self.dataloaders[k])
                else:
                    self._prefetchers[k] = None

    def _maybe_start_and_get_prefetcher(self, loop):
        if self.trainer_params.use_prefetch and self._prefetchers[loop] is not None:
            if self._prefetchers[loop].finished:
                self._prefetchers[loop].re_init()
            self._prefetchers[loop].start()
            return self._prefetchers[loop]
        else:
            return None

    def _maybe_join_prefetchers(self):
        for k, v in self._prefetchers.items():
            if v is not None:
                self.logger.info(f"Joining prefetcher for {k}")
                v.finish()
                v.join()

    def _maybe_join_prefetcher(self, loop):
        if self._prefetchers[loop] is not None:
            self._prefetchers[loop].finish()
            self._prefetchers[loop].join()

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

    def _pp(self, func: Callable) -> Callable:
        """Alias for :meth:`_prepare_function`
        """
        return self._prepare_function(func)

    def _prepare_function(self, func: Callable) -> Callable:
        """Function to replace first argument as :class:`Trainer` instance
        It's applied to all the hooks.

        Args:
            func: The function to modify

        """
        if isinstance(func, partial):
            if hasattr(func.func, "__self__"):
                return func
            else:
                return partial(func.func, self, *func.args, **func.keywords)
        else:
            if hasattr(func, "__self__"):
                return func
            else:
                return partial(func, self)

    def _init_hooks(self):
        """Initialize all hooks.

        For the :class:`Trainer` each function in a hook must take an instance
        of :class:`trainer` itself as the first argument.

        """
        # CHECK: Why do we initialize super() so late?
        super().__init__(self.logger)
        self._hooks = {"post_init_hook": [self._pp(initialize_seed)],
                       "pre_resume_hook": [],
                       "post_resume_hook": [],
                       "pre_save_hook": [],
                       "post_save_hook": [],
                       "pre_eval_hook": [self._pp(pre_eval_log)],
                       "post_eval_hook": [self._pp(update_metrics)],
                       "pre_batch_hook": [self._pp(pre_batch_init_batch_vars)],
                       "post_batch_hook": [self._pp(post_batch_update_batch_vars)],
                       "pre_update_call_hook": [],
                       "post_update_call_hook": [],
                       "pre_training_hook": [self._pp(pre_train_log)],
                       "post_training_hook": [],
                       "pre_epoch_hook": [],
                       "post_epoch_hook": [*map(self._pp,
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
    def order(self) -> List[str]:
        """The order of execution of the pipeline"""
        pass

    @property
    def cmds(self) -> List[str]:
        return cmd.names

    @property
    def props(self) -> List[str]:
        """Return all properties of the instance including except hidden properties
        """
        return [x for x, y in self.__class__.__dict__.items()
                if isinstance(y, property) and x != "props"
                and not x.startswith("_")]

    @property
    def name(self) -> str:
        return self._name

    @property
    def checkpoint_name(self) -> str:
        return "_".join([self._checkpoint_prefix, self._name,
                         self._model.model_name, self.data["name"]])\
                  .replace(" ", "_")  # type: ignore

    @property
    def save_best_name(self) -> str:
        return "_".join([self._save_best_prefix, self._name,
                         self._model.model_name, self.data["name"]])\
                  .replace(" ", "_")  # type: ignore

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
            if data.get(x, None) and data[x] is not None:
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
            self.logger.warning(f"{func} SHOULD be an instance of {UpdateFunction}")
            # FIXME: This doesn't do anything right now
        elif not callable(func):
            err_msg = f"Not Callable {func}"
            self.logger.error(err_msg)
            raise AttributeError(err_msg)
        elif not hasattr(func, "train"):
            err_msg = f"Attribute train not in {func}"
            self.logger.error(err_msg)
            raise AttributeError(err_msg)
        elif not hasattr(func, "returns"):
            err_msg = f"Attribute returns not in {func}"
            self.logger.error(err_msg)
            raise AttributeError(err_msg)
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

    @property
    def extra_opts(self) -> Dict:
        return self._extra_opts

    def run_cmd(self, _cmd, *args, **kwargs):
        """Run command `_cmd` with `args` and `kwargs`.

        The commands are kept in :class:`Tag` `cmd` and the trainer
        instance is passed as first argument to them

        """
        cmd[_cmd](self, *args, **kwargs)

    def _init_metrics(self):
        self._metrics: Dict[str, Dict[str, Dict]] = {}
        for x in ["train", "val", "test"]:
            self._metrics[x] = {m: {} for m in self.trainer_params.metrics}
            self._metrics[x]["time"] = {}
        metrics = set(m for m in self.trainer_params.metrics)
        if isinstance(self.update_function, UpdateFunction):
            diff = set(metrics) - set(self.update_function.returns)
            if diff:
                err_msg = f"{diff} metrics are not returned by update_function"
                self.logger.error(err_msg)
                raise AttributeError(err_msg)

    @cmd
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
            raise AttributeError("Error. Trying to load from a different model\n" +
                                 "Use force resume if you want to do this")
        if saved_state["data_name"] != self.data["name"]:
            raise AttributeError("Error. Trying to load a different dataset\n" +
                                 "Use force resume if you want to do this")
        self._epoch = saved_state["epoch"]
        self.logger.info(f"Resuming from checkpoint at end of epoch {self.epoch+1}")
        if isinstance(self.model, torch.nn.parallel.DataParallel):
            self.model.module.load_state_dict(saved_state['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state['model_state_dict'])
        if hasattr(self, "load_optimizer"):
            self.logger.info("Will load optimizer from custom \"load_optimizer\"")
            optimizer_state = self.load_optimizer(saved_state['optimizer_state_dict'])
        else:
            optimizer_state = saved_state['optimizer_state_dict']
        self.optimizer.load_state_dict(optimizer_state)
        if hasattr(self, "load_extra"):
            self.logger.info("Loading extra state")
            self.load_extra(self, saved_state)
        self._metrics = saved_state["metrics"]
        for k, v in self.trainer_params.dict().items():
            if k in saved_state["params"] and v != saved_state['params'][k]:
                self.logger.warning(f"Param {k} is overwritten from args. " +
                                    f"Old value {saved_state['params'][k]}, new value {v}")
                saved_state["params"][k] = self.trainer_params.dict()[k]
        self.trainer_params = saved_state["params"]
        for k, v in self.extra_opts.items():
            if k in saved_state["extra_opts"] and v != saved_state["extra_opts"][k]:
                self.logger.warning(f"Extra opt {k} is overwritten from args. " +
                                    f"Old value {saved_state['extra_opts'][k]}, new value {v}")
            elif k not in saved_state["extra_opts"]:
                self.logger.warning(f"New option {k} in extra_opts.")
        self.run_hook("post_resume_hook")

    def _save(self, name):
        self.run_hook("pre_save_hook")
        save_name = name if name.endswith(".pth") else name + ".pth"
        if hasattr(self, "save_optimizer") and hasattr(self, "load_optimizer"):
            self.logger.info("Will save according to custom \"save_optmizer\"")
            optimizer_state_dict = self.save_optimizer()
        else:
            optimizer_state_dict = self.optimizer.state_dict()
        if hasattr(self, "save_model") and hasattr(self, "load_model"):
            self.logger.info("Will save according to custom \"save_model\"")
            model_state_dict = self.save_model()
        else:
            if isinstance(self.model, torch.nn.parallel.DataParallel):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
        if hasattr(self, "save_extra"):
            extra_saves = self.save_extra(self)
        else:
            extra_saves = {}
        if extra_saves:
            self.logger.info(f"Will save extra state for {extra_saves.keys()}")
        torch.save({'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer_state_dict,
                    'metrics': self.metrics,
                    "epoch": self.epoch,
                    'data_name': self.data["name"],
                    'model_name': self._model.model_name,
                    'params': self.trainer_params.__dict__,
                    'extra_opts': self.extra_opts,
                    **extra_saves},
                   self.savedir.joinpath(save_name))
        self.run_hook("post_save_hook")

    def _next_batch(self, loop_name):
        pass

    def _eval(self, eval_loop_name, debug=False):
        loader = self.dataloaders[eval_loop_name]
        self.run_hook_with_args("pre_eval_hook", loop=eval_loop_name)
        total_iters = len(loader)
        prefetcher = self._maybe_start_and_get_prefetcher(eval_loop_name)
        if prefetcher is None:
            it = loader.__iter__()
        i = 0
        try:
            while True:
                with self.timer:
                    if prefetcher is not None:
                        batch = self._prefetchers[eval_loop_name].get()
                    else:
                        batch = it.__next__()
                self.eval_one_batch(eval_loop_name, i, total_iters, batch_time=self.timer.time,
                                    batch=batch)
                i += 1
        except StopIteration:
            self._maybe_join_prefetcher(eval_loop_name)
        self.run_hook_with_args("post_eval_hook", loop=eval_loop_name)

    @cmd
    def validate(self):
        self._eval("val")

    @cmd
    def test(self):
        self._eval("test")

    @cmd
    def eval_one_batch(self, val_or_test, batch_num, total_iters, batch_time, batch):
        self.run_hook_with_args("pre_batch_hook", loop=val_or_test,
                                batch_num=batch_num)
        with torch.no_grad():
            with self.timer:
                self.update_function.train = False
                self.model.eval()
                with torch.no_grad():
                    retval = self.update_function(batch=batch, criterion=self.criterion,
                                                  model=self.model, optimizer=self.optimizer,
                                                  trainer=self, batch_num=batch_num)
        retval.update(self.timer.as_dict)
        self.run_hook_with_args("post_batch_hook", loop=val_or_test, retval=retval,
                                batch_num=batch_num, total_iters=total_iters,
                                batch_time=batch_time)

    @cmd
    def train_one_batch(self, batch_num, total_iters, batch_time, batch):
        self.run_hook_with_args("pre_batch_hook", loop="train",
                                batch_num=batch_num)
        with self.timer:
            self.update_function.train = True
            self.model.train()
            retval = self.update_function(batch=batch, criterion=self.criterion,
                                          model=self.model, optimizer=self.optimizer,
                                          trainer=self, batch_num=batch_num)
        retval.update(self.timer.as_dict)
        self.run_hook_with_args("post_batch_hook", loop="train", retval=retval,
                                batch_num=batch_num, total_iters=total_iters,
                                batch_time=batch_time)

    @cmd
    def run_one_epoch(self, testing=False, num_iters=2):
        self.logger.info(f"Training epoch {self.epoch+1}")
        self.run_hook("pre_epoch_hook")
        total_iters = len(self.dataloaders["train"])
        prefetcher = self._maybe_start_and_get_prefetcher("train")
        if prefetcher is None:
            it = self.dataloaders["train"].__iter__()
        i = 0
        try:
            while True:
                with self.timer:
                    if prefetcher is not None:
                        batch = prefetcher.get()
                    else:
                        batch = it.__next__()
                self.train_one_batch(i, total_iters, batch_time=self.timer.time,
                                     batch=batch)
                i += 1
                if testing:
                    if i == num_iters:
                        break
        except StopIteration:
            self._maybe_join_prefetcher("train")
        self.run_hook("post_epoch_hook")

    @cmd
    def train(self):
        """Start training
        """
        self.run_hook("pre_training_hook")
        while self.epoch < self.trainer_params.max_epochs:
            self.run_one_epoch()
        self.run_hook("post_training_hook")
        self.logger.info('Finished training')

    @cmd
    def start(self):
        """Start training after trying to resume.
        """
        self.try_resume()
        self.train()

    @cmd
    def test_loops(self, num_iters=2):
        """Test all the loops to see if everything works
        Run only two batches.
        """
        self.logger.warning("After testing the loops, " +
                            "please make sure to reinitialize the dataloader and model, " +
                            "as the weights and next batch etc would have changed.")
        self.run_hook("pre_training_hook")
        self.run_one_epoch(testing=True, num_iters=num_iters)
        self.try_resume()
        self.run_one_epoch(testing=True)
        # TODO: Add this in later maybe
        # self.post_testing_cleanup()
        self.run_hook("post_training_hook")
        self.logger.info('Finished testing the loops')
