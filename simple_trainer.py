import re
import os
import torch
from torch.utils.data import DataLoader
# from torch import Tensor
# from typing import Union, List


def rename_or_set_default(names, namespace, name, default):
    for _name in names:
        if _name in namespace:
            namespace.__dict__[name] = namespace.__dict__[_name]
            return
    namespace.__dict__[name] = default


class Trainer:
    def __init__(self, args, model, data, dataloader_params, steps, logger=None):
        """Initializes the :class: `trainer.Trainer` object.

        :param args: args is a :class: `types.SimpleNamespace`
        :param model: model which is a :class: `torch.nn.Module`
        :param train_loader: a train data loader usually :class: `torch.utils.data.Dataloader`
        :param val_loader: a validation data loader usually :class: `torch.utils.data.Dataloader`
        :param test_loader: a test data loader usually :class: `torch.utils.data.Dataloader`
        :param logger: an instance of :class: `logging.Logger`
        :returns: None
        :rtype: None
        """
        assert logger is not None, "logger cannot be None"
        self.logger = logger
        self.args = self._sanity_check(args)
        self._data = data
        self._dataloader_params = dataloader_params
        self.steps = steps
        assert all(callable(x) for x in self.steps.values())
        # TODO: Make this more generic
        self.save_on = 'val'
        self.save_var = 'losses'
        assert self.save_on in {'test', 'val'}
        assert self.save_var in ['losses', 'accuracies']
        self.logger.debug("Initializing trainer with args %s" % str(self.args))
        self._init_model(model)
        self._init_dataloaders()
        self.criterion = args.criterion
        self.optimizer = self._get_optimizer(args.optimizer)
        # TODO: Make this more generic
        self.losses = []
        self.accuracies = []
        self.prev_len_var_check = 0
        self.epoch = 0
        if self.args.resume_checkpoint:
            assert not self.args.init_weights and not self.args.resume_best
            self._resume_path = os.path.join(self._savedir, self.args.checkpoint_name)
            assert os.path.isfile(self._resume_path) and os.path.exists(self._resume_path)
            self.logger.info("Trying to resume from checkpoint")
            self._resume()
        elif self.args.resume_from_weights:
            assert not self.args.init_weights and not self.args.resume_best and not self.args.resume_checkpoint
            assert os.path.exists(self.args.resume_from_weights)
            self._resume_path = self.args.resume_from_weights
            self.logger.info("Trying to resume from given weights")
            self._resume()
            # TODO: resume_best assumes "val_acc" in file name
        elif self.args.resume_best:
            assert os.path.exists(os.path.join(self._savedir)) and os.listdir(self._savedir)
            # Assuming for each training session the directory is the same
            save_files = os.listdir(self._savedir)
            # FIXME
            if "checkpoint.pth" in save_files:
                save_files.remove("checkpoint.pth")
            save_files = [(f, re.search("val_.....", f).group()) for f in save_files]
            save_files.sort(key=lambda x: x[1])
            self._resume_path = os.path.join(self._savedir, save_files[-1][0])
            self._resume()
        elif self.args.init_weights:
            assert not self.args.resume_checkpoint and not self.args.resume_best
            assert os.path.exists(self.args.init_weights) and os.path.isfile(self.args.init_weights)
            self.logger.info("Trying to load torch save directly into model")
            self._load_init_weights()

    def _sanity_check(self, args):
        assert 'lr' in args.__dict__ and args.lr > 0, "Learning rate must exist and be > 0"
        assert 'momentum' in args.__dict__
        assert 'print_freq' in args.__dict__
        rename_or_set_default(['savedir', 'save_dir'], args, 'savedir', 'savedir')
        rename_or_set_default(['optim', 'optimizer'], args, 'optimizer', 'sgd')
        rename_or_set_default(['epochs', 'num_epochs', 'max_epochs'], args, 'max_epochs', 100)
        rename_or_set_default(['drop_ratio', 'do'], args, 'drop_ratio', None)
        rename_or_set_default(['fc_drop_ratio', 'fc_do'], args, 'fc_drop_ratio', None)
        rename_or_set_default(['checkpoint_path', 'checkpoint', 'checkpoint_name'],
                              args, 'checkpoint_path', 'checkpoint.pth')
        rename_or_set_default(['resume_best'], args, 'resume_best', False)
        rename_or_set_default(['resume_checkpoint'], args, 'resume_checkpoint', False)
        rename_or_set_default(['init_weights', 'weights'], args, 'init_weights', None)
        if not os.path.exists(args.savedir):
            os.mkdir(args.savedir)
        self._savedir = args.savedir
        self.max_epochs = args.max_epochs
        return args

    def _init_model(self, model):
        self.model = model
        if not self.args.cuda or not self.args.gpus or self.args.gpus == "cpu":
            self.device = torch.device("cpu")
        elif len(self.args.gpus) == 1:
            self.device = torch.device("cuda:%d" % self.args.gpus[0])
        else:
            self.device == "parallel"
        if self.device == "parallel":
            self.model = self.model.cuda()
        else:
            self.model = self.model.to(self.device)

    def _init_dataloaders(self):
        assert all([x in self._data for x in ["train", "val", "test"]])
        if not self._data["train"]:
            raise AttributeError
        self.train_loader = DataLoader(self._data["train"], **self._dataloader_params["train"])
        if self._data["val"]:
            self.val_loader = DataLoader(self._data["val"], **self._dataloader_params["val"])
        else:
            self.val_loader = None
        if self._data["test"]:
            self.test_loader = DataLoader(self._data["test"], **self._dataloader_params["test"])
        else:
            self.test_loader = None

    def _load_init_weights(self):
        self.logger.warn("Warning! Loading directly to model")
        self.model.load_state_dict(torch.load(self.args.init_weights).state_dict())

    def _resume(self):
        self.logger.info("Resuming from %s" % self._resume_path)
        saved_state = torch.load(self._resume_path)
        assert saved_state['model'] == self.model.__class__.__name__, "Error. Trying to load from a different model"
        assert 'model_state_dict' in saved_state
        assert 'optimizer_state_dict' in saved_state
        # TODO: Make this more generic
        assert 'accuracies' in saved_state
        assert 'losses' in saved_state
        assert 'params' in saved_state
        self.model.load_state_dict(saved_state['model_state_dict'])
        self.optimizer.load_state_dict(saved_state['optimizer_state_dict'])
        self.optimizer.param_groups[0]['lr'] = self.args.lr
        # TODO: Make this more generic
        self.accuracies = saved_state['accuracies']
        self.losses = saved_state['losses']
        self.epoch = saved_state['params']['epoch']
        self.args.drop_ratio = saved_state['params']['drop_ratio']
        self.args.fc_drop_ratio = saved_state['params']['fc_drop_ratio']
        self.logger.debug("Learning Rate %s" % str(self.optimizer.param_groups[0]['lr']))
        self.validate()

    def _get_optimizer(self, name):
        # TODO: Need additional optimizers. Check should only be if it's
        #       a subclass of torch.optim.Optimizer
        if name.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif name.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters())

    def _save(self, save_name_or_dict):
        if isinstance(save_name_or_dict, dict):
            save_name = '__'.join(['_'.join([a, str(b)]) for a, b in save_name_or_dict.items()]) + '.pth'
        elif isinstance(save_name_or_dict, str):
            save_name = save_name_or_dict if save_name_or_dict.endswith('.pth') else save_name_or_dict + '.pth'
        else:
            raise AttributeError
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # TODO: Make this more generic
                    'accuracies': self.accuracies,
                    'losses': self.losses,
                    'model': self.model.__class__.__name__,
                    'params': {'epoch': self.epoch,
                               'drop_ratio': self.args.drop_ratio,
                               'fc_drop_ratio': self.args.fc_drop_ratio}},
                   os.path.join(self._savedir, save_name))

    # TODO: Make this more generic
    def check_and_save(self):
        def _check_predicate(cur, prev):
            if metavar == 'losses':
                return cur < prev
            elif metavar == 'accuracies':
                return cur > prev
            else:
                raise ValueError
        assert self.save_on in ['train', 'val']
        save_on = self.save_on
        metavar = self.save_var  # accuracies or losses
        # TODO: Make this more generic
        check_losses = [l[2] for l in self.losses if l[0] == save_on]
        check_accuracies = [l[2] for l in self.accuracies if l[0] == save_on]
        # FIXME
        # train_accuracies = [l[2] for l in self.accuracies if l[0] == 'train']
        train_losses = [l[2] for l in self.losses if l[0] == 'train']
        if metavar == 'accuracies':
            _var_to_check = check_accuracies
            _check_fn = max
        elif metavar == 'losses':
            _var_to_check = check_losses
            _check_fn = min
        else:
            raise ValueError
        if _var_to_check and len(_var_to_check) > self.prev_len_var_check:
            self.prev_len_var_check += 1
            cur_train_loss = train_losses[-1]
            # if train_accuracies:
            #     cur_train_acc = train_accuracies[-1]
            _cur_check_var = _var_to_check[-1]
            save_name_dict = dict([('model', self.model.__class__.__name__),
                                   ('epoch', self.epoch),
                                   ('drop_ratio', self.args.drop_ratio),
                                   ('fc_drop_ratio', self.args.fc_drop_ratio),
                                   ('train_loss', cur_train_loss),
                                   ('_'.join([save_on, 'loss']), check_losses[-1]),
                                   # TODO: Make this more generic
                                   # ('train_acc', cur_train_acc),
                                   # ('_'.join([save_on, 'acc']), check_accuracies[-1]),
                                   ('train_batch_size', self._dataloader_params["train"]["batch_size"])])
            if len(_var_to_check) == 1:
                self.logger.info('Saving model as ' + str(save_name_dict))
                self._save(save_name_dict)
            elif len(_var_to_check) >= 2:
                _prev_check_var = _check_fn(_var_to_check[:-1])
                self.logger.info('Checking %s %s: cur = %f, prev = %f' %
                                 (save_on, metavar, _cur_check_var, _prev_check_var))
                if _check_predicate(_cur_check_var, _prev_check_var):
                    self.logger.info('Saving model as ' + str(save_name_dict))
                    self._save(save_name_dict)
                else:
                    self.logger.info("%s didn't improve" % metavar)
        else:
            self.logger.info("No %s to check" % metavar)

    def _test(self, val_test, debug=False):
        if val_test == "val":
            self.logger.info("Validating the model")
            loader = self.val_loader
        elif val_test == "test":
            self.logger.info("Testing the model")
            loader = self.test_loader
        total = 0
        loss = 0
        for i, batch in enumerate(loader):
            retval = self.steps[val_test](self, batch)
            loss += retval["loss"]
            total += retval["total"]
            if debug:
                print(retval["outputs"].cpu().numpy(), retval["targets"].cpu().numpy())
                import ipdb
                ipdb.set_trace()
        self.losses.append((val_test, self.epoch, float(loss)))
        self.logger.info('Average loss for of the network on %d %s images: %f' %
                         (total, val_test, loss/len(loader)))

    def validate(self, debug=False):
        self._test("val", debug)

    def test(self):
        self._test("test")

    def _batch_end(self, retval, batch_num):
        self.running_loss += retval["loss"]
        if batch_num % self.args.print_freq == max(0, self.args.print_freq - 1):
            self.logger.info('[%d, %5d] loss: %.3f. %f percent epoch done' %
                             (self.epoch + 1, batch_num + 1,
                              self.running_loss,
                              batch_num / len(self.train_loader) * 100))
            self.running_loss = 0.0
        if self.model_name == "classifier":
            return retval["total"], retval["loss"], retval["accuracy"]
        else:
            return retval["total"], retval["loss"]

    def _log_epoch_end_loss(self, total, epoch_loss, accuracy):
        self.losses.append(('train', self.epoch, epoch_loss))
        if self.model._model_name == "classifier":
            self.accuracies.append(('train', self.epoch, accuracy))
        self.losses.append(('train', self.epoch, epoch_loss))
        self.logger.info('Average loss of the network on %d images: %f'
                         % (total, epoch_loss/len(self.train_loader)))
        if self.model._model_name == "classifier":
            self.logger.info('Average accuracy of the network on %d images: %f'
                             % (total, accuracy/len(self.train_loader)))

    def _validate_and_test(self):
        if self.val_loader is not None:
            self.validate()
        else:
            self.logger.info("No val loader. Skipping")
        if self.epoch % 5 == 4:
            if self.test_loader is not None:
                self.test(self.epoch)
            else:
                self.logger.info("No test loader. Skipping")
        if self._anneal:
            self.anneal()

    def _epoch_end_save(self, epoch_loss, old_epoch_loss):
        self.logger.info("Saving checkpoint to %s" % self.args.checkpoint_path)
        self.logger.info("Epoch loss and prev epoch loss: %f %f" % (epoch_loss, old_epoch_loss))
        self._save(self.args.checkpoint_path)
        self.check_and_save()

    def train(self):
        self.validate(True)
        self.logger.debug("Beginning training")
        self.logger.debug("Total number of batches %d" % len(self.train_loader))
        total = 0
        epoch_loss = 0.0
        old_epoch_loss = 0.0
        accuracy = 0.0
        while self.epoch < self.args.max_epochs:
            self.running_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                retval = self.steps["train"](self, batch)
                if self._model_name == "classifier":
                    count, loss, batch_accuracy = self._batch_end(retval)
                    accuracy += batch_accuracy
                else:
                    count, loss = self._batch_end(retval)
                epoch_loss += loss
                total += count
            self._log_epoch_end_loss(total, epoch_loss, accuracy)
            self._validate_and_test()
            total = 0.0
            self._epoch_end_save(epoch_loss, old_epoch_loss)
            old_epoch_loss = epoch_loss
            epoch_loss = 0.0
            self.epoch += 1
        self.logger.info('finished training')

    # TODO: Should take into consideration a number of prev epochs and losses
    def anneal(self, cur_loss, old_loss):
        if self.epoch > 20 and (old_loss - cur_loss < (.01 * old_loss)):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= .9
            self.logger.info("Annealing...")

    def sample(self):
        raise NotImplementedError

    # FIXME: The syntax for typing hints wasn't correct. Have to figure out
    #        how to include types for torch.Tensor
    # # TODO: Sample should not return Tensor but instead visualizable values
    # #       It's hard to quantify what a "visualizable" value is.
    # def sample(self, inputs: Union[List[Union[int, float]],
    #                                Tensor[Union[int, float]]]) -> Tensor[Union[int, float]]:
    #     """Skeleton sample function, see PEP 484, and mypy for how the typing hint
    #     syntax works. Docstring is automatically generated.

    #     :param inputs: 
    #     :returns: 
    #     :rtype:

    #     """
    #     raise NotImplementedError
