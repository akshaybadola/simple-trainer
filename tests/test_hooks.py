import pytest
import re
from simple_trainer import functions


def test_trainer_should_log_post_batch_progress_after_adding_hook(capsys, trainer_with_mnist):
    trainer = trainer_with_mnist
    desc = trainer.describe_hook("post_batch_hook")
    if not any("post_batch_progress" in x for x in desc):
        trainer.add_to_hook_at_end("post_batch_hook", functions.post_batch_progress)
    train_iter = trainer._dataloaders["train"].__iter__()
    test_iter = trainer._dataloaders["test"].__iter__()
    train_batch = train_iter.__next__()
    test_batch = test_iter.__next__()
    trainer.train_one_batch(1, train_batch)
    captured = capsys.readouterr()
    messages = [x for x in captured.err.split("\n") if x.startswith("Message")]
    assert re.match(r".*Progress for epoch.+?batch.*?\\n", messages[-1])
    assert re.match(r".*?\\nAverage metric loss for train for last [0-9]+ batches.*", messages[-1])
    trainer.eval_one_batch("test", 1, test_batch)
    captured = capsys.readouterr()
    messages = [x for x in captured.err.split("\n") if x.startswith("Message")]
    assert re.match(r".*Progress for epoch.+?batch.*?\\n", messages[-1])
    assert re.match(r".*?\\nAverage metric loss for test for last [0-9]+ batches.*", messages[-1])
