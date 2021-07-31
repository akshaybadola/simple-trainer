import pytest
from functools import partial


def test_create_hook(hooks):
    hooks.create_hook("hook_a", [])
    hooks.create_hook("hook_b")
    assert "hook_a" in hooks._hooks
    assert "hook_b" in hooks._hooks


def test_add_remove_hook(hooks):
    def func_a(self):
        print("Does something")

    def func_b(self):
        print(f"{self}")

    def func_c(self, arg_a, arg_b):
        print(arg_a, arg_b)

    pf = partial(func_c, "a")
    hooks.create_hook("hook_a", [])
    hooks.create_hook("hook_b")
    hooks.add_hook("hook_a", func_a)
    hooks.add_hook("hook_a", func_b)
    assert hooks.describe_hook("hook_a") == ["func_b, {}", "func_a, {}"]
    hooks.add_hook("hook_b", func_a)
    hooks.add_hook_after("hook_b", func_b, "func_a")
    assert hooks.describe_hook("hook_b") == ["func_a, {}", "func_b, {}"]
    hooks.add_hook_at_end("hook_a", func_a)
    hooks.add_hook_at_beginning("hook_a", func_a)
    assert hooks.describe_hook("hook_a") == ["func_a, {}", "func_b, {}", "func_a, {}", "func_a, {}"]
    # add partial
    hooks.add_hook("hook_b", pf)
    assert hooks.describe_hook("hook_b") == ["partial func_c, ('a',), {}",
                                             "func_a, {}", "func_b, {}"]
    # remove first
    hooks.remove_hook("hook_a", "func_a")
    assert hooks.describe_hook("hook_a") == ["func_b, {}", "func_a, {}", "func_a, {}"]
    # remove first partial
    hooks.remove_hook("hook_b", "func_c")
    assert hooks.describe_hook("hook_b") == ["func_a, {}", "func_b, {}"]
