from typing import List, Dict, Callable, Union, Optional
import abc
import types
import logging
from functools import partial
from contextlib import ExitStack
from common_pyutil.functional import first, maybe_then


def partial_or_func_name(x: Callable, describe: bool = False):
    return maybe_then(x, [partial, types.FunctionType],
                      [lambda x: ("partial " if describe else "") + x.func.__name__,
                       lambda x: x.__name__])


class Hooks(abc.ABC):
    """A simple class implmentation for flow based programming

    Hooks are named points in an execution pipeline at which functions can be
    added and removed dynamically. Hooks should be:

        1. Configurable
        2. Inspectable
        3. Modifiable

    Unlike a standard pipeline, hooks can be added and removed by user, and
    functions to the hook can be altered programmatically and interactively
    through the publicly exposed API.

    """
    def __init__(self, logger: logging.Logger):
        """Initialize with a logger.

        Args:
            logger: A logger for logging

        """
        self._hooks: Dict[str, List[Callable]] = {}
        self.logger = logger

    def __iter__(self):
        return self._hooks.keys().__iter__()

    def create_hook(self, hook_name: str, funcs: List[Callable] = []):
        """Create a `hook` with given name

        Args:
            hook_name: Name of the hook to create
            funcs: List of functions to initialize the hook.

        Each function can be called with or without arguments. The arguments
        must be keyword arguments. The hook must process the kwargs as it
        receives. This provides for the most flexibility without inspecting the
        code. It's upto the functions on how they will process the keyword
        arguments.

        Example:
            from types import SimpleNamespace

            def some_function(**kwargs):
                processable_args = ["arg1", "arg2", "arg3"]
                args_dict = {arg: kwargs.get(arg, None) for arg in processable_args}
                args = SimpleNamespace(**args_dict)

                if any([args_dict[arg] is None for arg in processable_args]):
                    # maybe raise error or catch error and don't
                    # do anything with a warning

        """
        if hook_name in self._hooks:
            raise AttributeError(f"Hook {hook_name} already exists")
        else:
            self._hooks[hook_name] = funcs

    def delete_hook(self, hook_name: str):
        """Delete a named `hook` from Hooks

        Args:
            hook_name: Name of the hook to delete

        """
        if hook_name not in self._hooks:
            raise AttributeError(f"No such hook {hook_name}")
        else:
            self._hooks.pop(hook_name)

    @abc.abstractmethod
    def _prepare_function(self, func: Callable) -> Callable:
        """Prepare a function to be added to a hook.

        When any function is added to a hook, it's transformed by with this
        function.  This has to be overridden. E.g., in the example below, each
        function added to any hook is called with `self` as the first argument.

        Example:
            class Pipeline(Hooks):
                def __init__(self):
                    pass

                def _prepare_function(self, func):
                    return partial(func, self)

        Or if you'd like to keep the :class:`Hooks` instance separate.

            class MyHooks(Hooks):
                def _prepare_function(self, func):
                    return func

            class Pipeline:
                def __init__(self):
                    self.hooks = MyHooks()

        """
        return func

    def check_func_args(self, func: Callable):
        if isinstance(func, partial):
            n_args = func.func.__code__.co_argcount
            if n_args != len(func.args) + len(func.keywords):
                raise AttributeError("Partial function must be fully specified")
        else:
            n_args = func.__code__.co_argcount
            if n_args:
                raise AttributeError("Function to the hook cannot take any arguments")

    def run_hook_with_contexts(self, hook_name: str, contexts: List, **kwargs):
        """Run a named hook with contexts

        Args:
            hook_name: Name of the hook
            contexts: contexts in which to run
            kwargs: Optional keyword arguments for hook

        """
        hook = self._get_hook(hook_name)
        if hook:
            with ExitStack() as stack:
                for con in contexts:
                    stack.enter_context(con)
                for func in hook:
                    func(**kwargs)

    def run_hook(self, hook_name: str):
        """Run a named hook.

        Args:
            hook_name: Name of the hook

        """
        hook = self._get_hook(hook_name)
        if hook:
            for func in hook:
                func()

    def run_hook_with_args(self, hook_name: str, **kwargs):
        """Run a named hook with arguments.

        Only keyword arguments are allowed. Therefore, for all the functions in
        the hook can accept an arbitrary number of arguments and the functions
        can check and choose the relevant arguments.

        Args:
            hook_name: Name of the hook
            kwargs: keyword arguments only for the hook

        """
        hook = self._get_hook(hook_name)
        if hook:
            for func in hook:
                func(**kwargs)

    def add_to_hook(self, hook_name: str, func: Callable, position: Union[int, str] = 0):
        """Add function :code:`func` to hook_name with name `hook_name`.

        Args:
            hook_name: Name of the hook
            func: A function with a single argument
            position: Where to insert the hook_name. Defaults to front of list.

        If `position` is not given then it's added to the front of the list.

        """
        f_name = partial_or_func_name(func, True)
        func = self._prepare_function(func)
        self.check_func_args(func)
        if hook_name in self._hooks:
            self.logger.info(f"Adding {f_name} to {hook_name} at {position}")
            if position == "first":
                pos = 0
            elif position == "last":
                pos = len(hook_name)
            elif isinstance(position, int):
                pos = position
            else:
                raise ValueError(f"Unknown Value for position {position}")
            self._hooks[hook_name].insert(pos, func)

    def add_to_hook_at_end(self, hook_name: str, func: Callable):
        self.add_to_hook(hook_name, func, "last")

    def add_to_hook_at_beginning(self, hook_name: str, func: Callable):
        self.add_to_hook(hook_name, func, "first")

    def add_to_hook_before(self, hook_name: str, func: Callable, before_func: str):
        """Add function :code:`func` to hook with given name.

        Args:
            hook_name: Name of the hook
            func: A function with a single argument
            position: Where to insert the hook_name. Defaults to front of list.
        """
        f_name = partial_or_func_name(func, True)
        func = self._prepare_function(func)
        self.check_func_args(func)
        if hook_name in self._hooks:
            self.logger.info(f"Adding {f_name} to {hook_name} before {before_func}")
            names = [partial_or_func_name(x) for x in self._hooks[hook_name]]
            if before_func in names:
                pos = names.index(before_func)
                self._hooks[hook_name].insert(pos, func)
            else:
                raise ValueError(f"No such func {before_func}")

    def add_to_hook_after(self, hook_name: str, func: Callable, after_func: str):
        """Add function :code:`func` to hook with given name.

        Args:
            hook_name: Name of the hook_name
            func: A function with a single argument
            position: Where to insert the hook_name. Defaults to front of list.
        """
        f_name = partial_or_func_name(func, True)
        func = self._prepare_function(func)
        self.check_func_args(func)
        if hook_name in self._hooks:
            self.logger.info(f"Adding {f_name} to {hook_name} after {after_func}")
            names = [partial_or_func_name(x) for x in self._hooks[hook_name]]
            if after_func in names:
                pos = names.index(after_func) + 1
                self._hooks[hook_name].insert(pos, func)
            else:
                raise ValueError(f"No such func {after_func}")

    def remove_from_hook(self, hook_name: str, function_name: str):
        """Remove from named hook the named function.

        Args:
            hook_name: The name of the hook_name
            function_name: The name of the function to remove

        If there are multiple functions with same name, remove only the first
        one from the list.

        """
        hook = self._get_hook(hook_name)
        if hook:
            func = first(hook, lambda x: partial_or_func_name(x) == function_name)
            self._hooks[hook_name].remove(func)

    def remove_from_hook_at(self, hook_name: str, position: int):
        """Remove from named hook the function at position.

        Args:
            hook_name: The name of the hook_name
            position: The position at which the function to remove

        """
        hook = self._get_hook(hook_name)
        if hook:
            hook.pop(position)

    def _get_hook(self, hook_name: str) -> Optional[List[Callable]]:
        """Get hook with give name if it exists.

        Args:
            hook_name: Name of the hook

        """
        if hook_name in self.hooks:
            return self.hooks[hook_name]
        else:
            return None

    @property
    def hooks(self) -> Dict[str, List[Callable]]:
        """Return all the hooks
        """
        return self._hooks

    def describe_hook(self, hook_name: str) -> List[str]:
        """Describe the hook with given name

        Args:
            hook_name: Name of the hook

        For each function in the hook, if it's a regular function return a
        string representation of a tuple of:
            1. The function name
            2. The function annotations

        If it's a :class:`partial` function:
            1. The function name
            2. The function arguments
            3. The keyword arguments

        """
        hook = self._get_hook(hook_name)
        retval = []
        if hook:
            for x in hook:
                if isinstance(x, partial):
                    retval.append(f"{partial_or_func_name(x, True)}")
                else:
                    retval.append(f"{x.__name__}")
        return retval
