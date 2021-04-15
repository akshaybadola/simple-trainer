from typing import Union, List, Optional, Any, Callable, Iterable, Dict
from pathlib import Path
from pydantic import BaseModel, validator
from torch.utils.data.distributed import DistributedSampler


def gpus_must_evaluate_to_list_of_int(v: Union[List[int], int, str, None]) ->\
        List[int]:
    if not isinstance(v, int) and not v:
        return []
    elif isinstance(v, str):
        try:
            retval = [*map(int, v.split(","))]
            return retval
        except Exception:
            raise ValueError(f"Unable to parse the gpus str {v}")
    elif isinstance(v, int):
        if v >= -1:
            return [v]
        else:
            return []
    else:
        return v


class DDPParams(BaseModel):
    backend: str
    init_method: str
    world_size: int
    num_gpus: int
    node_rank: int
    sampler: DistributedSampler

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class TrainerParams(BaseModel):
    """A config for parameters :class:`Trainer`

    Args:
        gpus: A list or comma separated string of gpus
        cuda: Whether to use cuda (gpus) or not
        seed: Seed with which torch will be initialized
        resume_best: Resume from the previously best state
        resume_checkpoint: Resume from given weights
        resume: Whether to resume or not.
        metrics: A list of names of which values to track
        val_frequency: How often to validate terms of `epoch`
        log_frequency: How often to log terms of `epoch`
        test_frequency: How often in terms of `epoch` to run the test loop
        dump_frequency: How often in terms of `epoch` to dump the trainer state
        max_epochs: Maximum epochs to train

    The behaviour of resuming from a previous state depends on both :code:`resume` and
    the params given.  If however, :code:`resume` is :code:`True` then :code:`resume_best` is
    checked first and :code:`trainer._resume_path` is set to that. Otherwise if
    :code:`resume_checkpoint` is given, then the state (including model weights) is
    resumed from there.

    Otherwise we resume from the last checkpoint.

    """
    gpus: Union[List[int], int, str, None]
    cuda: bool
    seed: int
    resume_checkpoint: Optional[Path]
    resume_best: Optional[bool]
    resume: bool
    metrics: List[str]
    val_frequency: Optional[int] = 1
    log_frequency: Optional[int] = 5
    test_frequency: Optional[int] = 5
    dump_frequency: Optional[int] = 5
    max_epochs: int
    save_best_on: Optional[str]
    save_best_by: Optional[str]

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    _validate_gpus = validator("gpus", allow_reuse=True)(gpus_must_evaluate_to_list_of_int)

    @validator("resume_checkpoint")
    def resume_checkpoint_must_be_path_and_exist(cls, v: Union[Path, str, None]) -> Optional[Path]:
        if not v:
            return None
        else:
            if Path(v).exists():
                if Path(v).is_file():
                    return Path(v)
                else:
                    raise AttributeError(f"Path {v} is not a file")
            else:
                raise AttributeError(f"Path {v} doesn't exist")

    @validator("resume_best")
    def only_one_of_resume_best_and_resume_checkpoint_can_be_given(cls,
                                                                   v: bool,
                                                                   values) -> bool:
        if "resume_checkpoint" in values and values["resume_checkpoint"]:
            raise ValueError("only one of resume_checkpoint or resume_best can be given")
        else:
            return v
