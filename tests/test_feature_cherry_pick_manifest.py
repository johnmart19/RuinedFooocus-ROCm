"""Validate prerequisite closure ordering without modifying a repository."""

import ast
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools.validate_feature_cherry_picks import cpu_test_code, dependency_order, module_exports


class FeatureDependencyTests(unittest.TestCase):
    def test_closure_omits_unrelated_features_and_orders_shared_dependencies_once(self):
        features = [
            dict(id="base"),
            dict(id="image", requires=["base"]),
            dict(id="chat", requires=["base"]),
            dict(id="api", requires=["image", "chat"]),
            dict(id="video", requires=["image"]),
        ]
        self.assertEqual(dependency_order(features, ["api"]), ["base", "image", "chat", "api"])

    def test_missing_dependency_and_cycle_are_rejected(self):
        with self.assertRaises(ValueError):
            dependency_order([dict(id="api", requires=["absent"])], ["api"])
        with self.assertRaises(ValueError):
            dependency_order([dict(id="a", requires=["b"]), dict(id="b", requires=["a"])], ["a"])

    def test_named_import_check_excludes_nested_and_comprehension_bindings(self):
        tree = ast.parse("""import os as operating_system
if True:
    PUBLIC = 1
def outer():
    local_only = 1
    def nested():
        pass
class Exported:
    class_member = 1
values = [temporary for temporary in []]
""")
        self.assertEqual(
            module_exports(tree), {"operating_system", "PUBLIC", "outer", "Exported", "values"}
        )

    def test_empty_test_discovery_cannot_pass(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        with (
            patch.object(unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()),
            patch.dict(sys.modules, {"torch": fake}),
            patch.dict(os.environ),
            patch.object(sys, "argv", ["cpu-check"]),
        ):
            with self.assertRaisesRegex(ValueError, "No CPU tests discovered"):
                exec(cpu_test_code(["test_missing.py"]), {})

    def test_cpu_checks_reject_an_initialized_gpu_before_discovery(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with (
            patch.object(unittest.defaultTestLoader, "discover") as discover,
            patch.dict(sys.modules, {"torch": fake}),
            patch.dict(os.environ),
            patch.object(sys, "argv", ["cpu-check"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "must not initialize a GPU"):
                exec(cpu_test_code(["test_example.py"]), {})
            discover.assert_not_called()

    def test_cpu_checks_prohibit_hardware_queries_during_discovery(self):
        for name in (
            "init",
            "is_available",
            "device_count",
            "current_device",
            "get_device_properties",
            "get_device_name",
            "get_device_capability",
            "mem_get_info",
        ):
            with self.subTest(query=name):
                fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))

                def discover(*args, **kwargs):
                    return getattr(fake.cuda, name)()

                with (
                    patch.object(unittest.defaultTestLoader, "discover", side_effect=discover),
                    patch.dict(sys.modules, {"torch": fake}),
                    patch.dict(os.environ),
                    patch.object(sys, "argv", ["cpu-check"]),
                ):
                    with self.assertRaisesRegex(RuntimeError, "prohibited in CPU feature checks"):
                        exec(cpu_test_code(["test_example.py"]), {})


if __name__ == "__main__":
    unittest.main()
