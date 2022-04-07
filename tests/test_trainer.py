import pytest
import re
from simple_trainer.helpers import ClassificationFunc
from simple_trainer import hook_functions


def test_trainer_init():
    pass


def test_trainer_should_assign_parameters_correctly():
    from simple_trainer.models import TrainerParams
    params = {"gpus": 0, "cuda": True, "seed": 1111, "resume": False, "metrics":
              ["loss", "accuracy"], "log_frequency": 1, "test_frequency": 1,
              "val_frequency": 1, "max_epochs": 100}
    trainer_params = TrainerParams(**params)


def test_trainer_update_function_should_not_set_function_with_wrong_signature(trainer):
    alt_func = lambda x: None
    trainer.update_function = alt_func
    assert trainer.update_function != alt_func
    alt_func = ClassificationFunc()
    trainer.update_function = alt_func
    assert trainer.update_function == alt_func


def test_trainer_add_and_remove_hook_should_work_correctly(trainer):
    def test_func(trainer):
        trainer._added_by_test_func = "test"
    trainer.add_to_hook("pre_resume_hook", test_func)
    assert trainer.describe_hook("pre_resume_hook")
    trainer.try_resume()
    assert trainer._added_by_test_func == "test"
    trainer.remove_from_hook_at("pre_resume_hook", 0)
    delattr(trainer, "_added_by_test_func")
    trainer.try_resume()
    assert not hasattr(trainer, "_added_by_test_func")


def test_trainer_one_batch_should_run_correctly(trainer_with_mnist):
    trainer = trainer_with_mnist
    train_iter = trainer._dataloaders["train"].__iter__()
    test_iter = trainer._dataloaders["test"].__iter__()
    train_batch = train_iter.__next__()
    trainer.train_one_batch(0, train_batch)
    test_batch = test_iter.__next__()
    trainer.eval_one_batch("test", 0, test_batch)
    test_batch = test_iter.__next__()
    trainer.eval_one_batch("test", 1, test_batch)
    trainer.run_hook_with_args("post_eval_hook", loop="test")


def test_trainer_should_log_post_batch_progress_at_correct_frequency(capsys, trainer_with_mnist):
    trainer = trainer_with_mnist
    train_iter = trainer._dataloaders["train"].__iter__()
    test_iter = trainer._dataloaders["test"].__iter__()
    trainer.trainer_params.log_frequency = 5
    desc = trainer.describe_hook("post_batch_hook")
    if not any("post_batch_progress" in x for x in desc):
        trainer.add_to_hook_at_end("post_batch_hook", hook_functions.post_batch_progress)
    j = 0
    while True:
        train_batch = train_iter.__next__()
        trainer.train_one_batch(j, train_batch)
        if j > 5:
            break
        j += 1
    j = 0
    while True:
        test_batch = test_iter.__next__()
        trainer.eval_one_batch("test", j, test_batch)
        if j > 5:
            break
        j += 1
    captured = capsys.readouterr()
    messages = [x for x in captured.err.split("\n") if x.startswith("Message")]
    assert any("Average metric loss for train for last 5 batches" in x for x in messages)
    assert any("Average metric loss for test for last 5 batches" in x for x in messages)
    assert any("Total time taken for last 5 batches" in x for x in messages)


def test_train_ddp():
    pass
