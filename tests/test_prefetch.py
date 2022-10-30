from simple_trainer.prefetch import Prefetcher


def test_prefetcher_should_init_and_join(trainer):
    trainer.trainer_params.use_prefetch = True
    trainer.trainer_params.max_epochs = 10
    trainer._maybe_init_prefetchers()
    trainer.try_resume()
    trainer.train()

