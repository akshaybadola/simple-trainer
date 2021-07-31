import pytest
import torch
from simple_trainer.trainer import Trainer
from simple_trainer.helpers import accuracy, ClassificationFunc


def test_trainer_init():
    pass


def test_trainer_params():
    from simple_trainer.models import TrainerParams
    params = {"gpus": 0, "cuda": True, "seed": 1111, "resume": False, "metrics":
              ["loss", "accuracy"], "log_frequency": 1, "test_frequency": 1,
              "val_frequency": 1, "max_epochs": 100}
    trainer_params = TrainerParams(**params)


def test_trainer_set_update_function(trainer):
    alt_func = lambda x: None
    trainer.update_function = alt_func
    assert trainer.update_function != alt_func
    alt_func = ClassificationFunc()
    trainer.update_function = alt_func
    assert trainer.update_function == alt_func


def test_trainer_add_remove_hook(trainer):
    def test_func(trainer):
        trainer._added_by_test_func = "test"
    trainer.add_hook("pre_resume_hook", test_func)
    assert trainer.describe_hook("pre_resume_hook")
    trainer.try_resume()
    assert trainer._added_by_test_func == "test"
    trainer.remove_hook_at("pre_resume_hook", 0)
    delattr(trainer, "_added_by_test_func")
    trainer.try_resume()
    assert not hasattr(trainer, "_added_by_test_func")


def test_trainer_one_batch(trainer_with_mnist):
    trainer = trainer_with_mnist
    train_batch = trainer._dataloaders["train"].__iter__().__next__()
    trainer.train_one_batch(0, train_batch)
    test_batch = trainer._dataloaders["test"].__iter__().__next__()
    trainer.eval_one_batch("test", 0, test_batch)
    test_batch = trainer._dataloaders["test"].__iter__().__next__()
    trainer.eval_one_batch("test", 1, test_batch)
    trainer.run_hook_with_args("post_eval_hook", loop="test")
