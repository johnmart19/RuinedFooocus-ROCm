"""Validate prerequisite closure ordering without modifying a repository."""
import ast
import unittest
from unittest.mock import patch
from tools.validate_feature_cherry_picks import cpu_test_code, dependency_order, module_exports


class FeatureDependencyTests(unittest.TestCase):
    def test_closure_omits_unrelated_features_and_orders_shared_dependencies_once(self):
        features = [dict(id='base'), dict(id='image', requires=['base']),
                    dict(id='chat', requires=['base']), dict(id='api', requires=['image', 'chat']),
                    dict(id='video', requires=['image'])]
        self.assertEqual(dependency_order(features, ['api']), ['base', 'image', 'chat', 'api'])

    def test_missing_dependency_and_cycle_are_rejected(self):
        with self.assertRaises(ValueError):
            dependency_order([dict(id='api', requires=['absent'])], ['api'])
        with self.assertRaises(ValueError):
            dependency_order([dict(id='a', requires=['b']), dict(id='b', requires=['a'])], ['a'])

    def test_named_import_check_excludes_nested_and_comprehension_bindings(self):
        tree = ast.parse('''import os as operating_system
if True:
    PUBLIC = 1
def outer():
    local_only = 1
    def nested():
        pass
class Exported:
    class_member = 1
values = [temporary for temporary in []]
''')
        self.assertEqual(module_exports(tree), {'operating_system', 'PUBLIC', 'outer', 'Exported', 'values'})

    def test_empty_test_discovery_cannot_pass(self):
        with patch.object(unittest.defaultTestLoader, 'discover', return_value=unittest.TestSuite()):
            with self.assertRaisesRegex(ValueError, 'No CPU tests discovered'):
                exec(cpu_test_code(['test_missing.py']), {})


if __name__ == '__main__':
    unittest.main()
