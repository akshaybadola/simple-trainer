import torch
import numpy as np


def initialize_seed(self):
    seed: int = self.trainer_params.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pre_val_log(self):
    self.logger.info("Testing the model")


def pre_batch_init_batch_vars(self, **kwargs):
    """Initialize :code:`trainer.Trainer.batch_vars`"""
    loop = kwargs["loop"]
    self._batch_running_loss = 0.0
    if not hasattr(self, "_batch_vars"):
        self._batch_vars = {loop: {}}
    elif loop not in self._batch_vars:
        self._batch_vars[loop] = {}


def post_batch_update_batch_vars(self, **kwargs):
    """Update the return values from a batch in `~trainer.Trainer.batch_vars`

    Args:
        loop: which loop is currently running
        retval: The value returned by the update function

    """
    loop, retval = kwargs["loop"], kwargs["retval"]
    for k, v in retval.items():
        if k not in self._batch_vars[loop]:
            self._batch_vars[loop][k] = []
        self._batch_vars[loop][k].append(v)


def post_batch_progress(self, **kwargs):
    """Display batch progress

    Args:
        i: batch number
        loop: which loop is currently running
    """
    i = kwargs["i"]
    loop = kwargs["loop"]
    lf = self.trainer_params.log_frequency
    if i % lf == (lf-1):
        log_str = []
        for k, v in self.batch_vars[loop].items():
            val = np.mean(v[i-lf:])
            log_str.append(f"Average metric {k} for {loop} for last {lf} batches: {val}")
        self.logger.info(f"Progress for epoch {self.epoch}, batch {i}\n" +
                         "\n".join(log_str))


def post_batch_log_running_loss(self, **kwargs):
    loop, batch_num = kwargs["loop"], kwargs["batch_num"]
    if batch_num % self.trainer_params.log_frequency ==\
       max(0, self.trainer_params.log_frequency - 1):
        self.logger.info('[%d, %5d] loss: %.3f. %f percent epoch done' %
                         (self.epoch + 1, batch_num + 1,
                          self._batch_running_loss,
                          batch_num / len(self.train_loader) * 100))
        self._batch_running_loss = 0.0


def post_epoch_reset_batch_vars(self):
    self._batch_vars = {}


def pre_eval_log(self, **kwargs):
    loop = kwargs["loop"]
    verb = "Testing" if loop == "test" else "Validating"
    self.logger.info(f"{verb} the model")


def pre_train_log(self):
    self.logger.debug("Beginning training")


def post_epoch_log(self):
    self.logger.debug(f"Finished epoch {self.epoch}")


def update_metrics(self, **kwargs):
    """Update the metrics in `trainer.Trainer`.

    For each loop's keys, we append all the variables which exist in values
    returned by the updated function.

    See also:
        :class:`models.UpdateFunction`.

    """
    loop = kwargs["loop"]
    self.logger.debug("Running update_metrics")
    for m in self._metrics[loop]:
        total = np.sum(self._batch_vars[loop]['total'])
        total_value = np.sum(self._batch_vars[loop][m])
        avg_value = total_value / total
        if m not in self._metrics[loop]:
            self._metrics[loop][m] = {}
        self._metrics[loop][m][self.epoch] = {"total": total_value, "average": avg_value}
        # self._metrics[loop][m][self.epoch]["average"] = avg_value
        self.logger.info(f'Total {loop} {m} of the network on ' +
                         f'{total} data instances is: {total_value}')
        self.logger.info(f'Average {loop} {m} of the network on ' +
                         f'{total} data instances is: {avg_value}')


def maybe_validate(self):
    self.logger.debug("Running maybe_validate")
    freq = self.trainer_params.val_frequency
    if self.epoch % freq == (freq - 1):
        if self.dataloaders["val"] is not None:
            self.validate()
        else:
            self.logger.info("No val loader. Skipping")


def maybe_test(self):
    self.logger.debug("Running maybe_test")
    freq = self.trainer_params.test_frequency
    if self.epoch % freq == (freq - 1):
        if self.dataloaders["test"] is not None:
            self.test()
        else:
            self.logger.info("No test loader. Skipping")


def maybe_anneal_lr(self, **kwargs):
    """Maybe anneal the learning rate.

    :code:`on`, :code:`after_epoch`, :code:`diff`, :code:`multiplier` must be
    present in kwargs.

    For a given metric :code:`on`, if the epoch is greater :code:`after_epoch`, and the
    difference between the metric for previous epoch is smaller than current
    epoch, within the fraction :code:`diff`, then we anneal the learning rate.

    The learning rate is multiplied by the :code:`multiplier` (should be below 1).

    """
    on = kwargs["on"]
    after_epoch = kwargs["after_epoch"]
    diff = kwargs["diff"]
    multiplier = kwargs["multiplier"]
    if self.epoch < after_epoch:
        return
    self.logger.debug(f"Running maybe_anneal_lr after epoch {self.epoch}")
    values = [(k, v["total"]) for k, v in self._metrics["train"][on].items()]
    values.sort(key=lambda x: x[0])
    old_val = values[-2][1]
    cur_val = values[-1][1]
    if old_val - cur_val < (diff * old_val):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= multiplier
        self.logger.info(f"Annealing running rate by {multiplier}")


def save_checkpoint(self):
    self.logger.info(f"Saving to {self.checkpoint_name}")
    self._save(self.checkpoint_name)


def save_best(self):
    save_on = self.trainer_params.save_best_on  # one of the loops
    save_by = self.trainer_params.save_best_by  # one of the metrics
    if not (save_on and save_by and save_on in self.dataloaders and
            save_by in self._metrics[save_on] and self.epoch > 1):
        self.logger.debug("Not Running save_best")
    else:
        values = self._metrics[save_on][save_by]
        if self._save_best_predicate(values):
            save_name = f"{self.save_best_name}_on_{save_on}_by_{save_by}"
            self.logger.debug(f"Saving Best to {save_name}")
            self._save(save_name)


def dump_state(self):
    freq = self.trainer_params.dump_frequency
    if self.epoch % freq == (freq - 1):
        self.logger.debug(f"Dumping state for epoch {self.epoch}")
        dump_name = self.checkpoint_name.replace(self._checkpoint_prefix, "dump") +\
            f"_epoch_{self.epoch:03}"
        self._save(dump_name)
