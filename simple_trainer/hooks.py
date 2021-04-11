import numpy as np


def pre_val_log(self):
    self.logger.info("Testing the model")


def pre_batch_init_batch_vars(self, loop):
    self.logger.debug("Running init_batch_vars")
    # loop in ["train", "val", "test"]
    self._batch_running_loss = 0.0
    if not hasattr(self, "_batch_vars"):
        self._batch_vars = {loop: {}}
    elif loop not in self._batch_vars:
        self._batch_vars[loop] = {}


def post_batch_update_batch_vars(self, loop, retval):
    self.logger.debug("Running update_batch_vars")
    for k, v in retval.items():
        if k not in self._batch_vars[loop]:
            self._batch_vars[loop][k] = []
        self._batch_vars[loop][k].append(v)


def post_batch_log_running_loss(self, loop, batch_num):
    self.logger.debug("Running log_running_loss")
    if batch_num % self.trainer_params.log_frequency ==\
       max(0, self.trainer_params.log_frequency - 1):
        self.logger.info('[%d, %5d] loss: %.3f. %f percent epoch done' %
                         (self.epoch + 1, batch_num + 1,
                          self._batch_running_loss,
                          batch_num / len(self.train_loader) * 100))
        self._batch_running_loss = 0.0


def pre_eval_log(self, loop):
    verb = "Testing" if loop == "test" else "Validating"
    self.logger.info(f"{verb} the model")


def pre_train_log(self):
    self.logger.debug("Beginning training")


def post_epoch_log(self):
    self.logger.debug(f"Finished epoch {self.epoch}")


def update_metrics(self, loop):
    self.logger.debug("Running update_metrics")
    for m in self._metrics[loop]:
        total = np.sum(self._batch_vars[loop]['total'])
        total_value = np.multiply(self._batch_vars[loop][m], self._batch_vars[loop]['total'])
        avg_value = np.sum(total_value) / total
        self._metrics[loop][m].append(avg_value)
        self.logger.info(f'Average {m} of the network on {total} data instances is: {avg_value}')


def maybe_validate(self):
    self.logger.debug("Running maybe_validate")
    freq = self.trainer_params.val_frequency
    if self.epoch % freq == (freq - 1):
        if self.val_loader is not None:
            self.validate()
    else:
        self.logger.info("No val loader. Skipping")


def maybe_test(self):
    self.logger.debug("Running maybe_test")
    freq = self.trainer_params.test_frequency
    if self.epoch % freq == (freq - 1):
        if self.test_loader is not None:
            self.test(self.epoch)
    else:
        self.logger.info("No test loader. Skipping")


def maybe_anneal_lr(self):
    self.logger.debug("Running maybe_anneal_lr")
    old_loss = self._metrics["train"]["loss"][-2]
    cur_loss = self._metrics["train"]["loss"][-1]
    anneal_epoch_after = getattr(self, "anneal_epoch_after", 20)
    anneal_loss_diff = getattr(self, "anneal_loss_diff", 0.01)
    anneal_multiplier = getattr(self, "anneal_multiplier", 0.9)
    if self.epoch > anneal_epoch_after and\
       (old_loss - cur_loss < (anneal_loss_diff * old_loss)):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= anneal_multiplier
        self.logger.info("Annealing running rate to {param_group['lr']}")


def save_checkpoint(self):
    self.logger.info("Saving checkpoint")
    self._save(self._checkpoint_name)


def save_best(self):
    self.logger.debug("Running save_best")
    save_on = self.trainer_params.save_best_on  # one of the loops
    save_by = self.trainer_params.save_best_by  # one of the metrics
    if not (save_on and save_by and save_on in self.dataloaders and
            save_by in self._metrics[save_on] and self.epoch > 1):
        return
    values = self._metrics[save_on][save_by]
    if self._save_best_predicate(values):
        self._save(f"{self._save_best_name}_on_{save_on}_by_{save_by}")
