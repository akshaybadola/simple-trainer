import pytest
from functools import partial


def test_create_hook(hooks):
    hooks.create_hook("hook_a", [])
    hooks.create_hook("hook_b")
    assert "hook_a" in hooks
    assert "hook_b" in hooks
    with pytest.raises(AttributeError):
        hooks.create_hook("hook_a")


def test_delete_hook(hooks):
    hooks.create_hook("hook_a", [])
    hooks.create_hook("hook_b")
    assert "hook_a" in hooks
    assert "hook_b" in hooks
    hooks.delete_hook("hook_a")
    assert "hook_a" not in hooks
    with pytest.raises(AttributeError):
        hooks.delete_hook("hook_c")


def test_add_remove_hook(hooks):
    def func_a():
        print("Does something")

    def func_b():
        print("Doesn't do anything")

    def func_c(arg_a, arg_b):
        print(arg_a, arg_b)

    pf = partial(func_c, "a", "b")
    hooks.create_hook("hook_a", [])
    hooks.create_hook("hook_b")
    hooks.add_hook("hook_a", func_a)
    hooks.add_hook("hook_a", func_b)
    assert hooks.describe_hook("hook_a") == ["func_b", "func_a"]
    hooks.add_hook("hook_b", func_a)
    hooks.add_hook_after("hook_b", func_b, "func_a")
    assert hooks.describe_hook("hook_b") == ["func_a", "func_b"]
    hooks.add_hook_at_end("hook_a", func_a)
    hooks.add_hook_at_beginning("hook_a", func_a)
    assert hooks.describe_hook("hook_a") == ["func_a", "func_b", "func_a", "func_a"]
    # add partial
    hooks.add_hook("hook_b", pf)
    assert hooks.describe_hook("hook_b") == ["partial func_c",
                                             "func_a", "func_b"]
    # remove first
    hooks.remove_hook("hook_a", "func_a")
    assert hooks.describe_hook("hook_a") == ["func_b", "func_a", "func_a"]
    # remove first partial
    hooks.remove_hook("hook_b", "func_c")
    assert hooks.describe_hook("hook_b") == ["func_a", "func_b"]
    hooks.add_hook_before("hook_b", pf, "func_b")
    assert hooks.describe_hook("hook_b") == ["func_a", "partial func_c", "func_b"]


def test_run_hook(hooks):
    def func_a(self):
        print("Does something")

    def func_b(self):
        print(f"{self}")

    def func_c(self, arg_a, arg_b):
        print(arg_a, arg_b)

    pf = partial(func_c, "a")

    # Add test for run hook, with args without args and with contexts
