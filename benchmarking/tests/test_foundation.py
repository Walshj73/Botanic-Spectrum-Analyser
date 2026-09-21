import os
import random
import unittest

from bsa_benchmark.core.registry import Registry
from bsa_benchmark.core.seeding import seed_everything


class RegistryTests(unittest.TestCase):
    def test_registry_rejects_duplicate_identifiers(self) -> None:
        registry = Registry("method")
        registry.register("example", lambda: object())
        self.assertEqual(("example",), registry.identifiers())
        with self.assertRaises(ValueError):
            registry.register("example", lambda: object())

    def test_unknown_identifier_lists_available_entries(self) -> None:
        registry = Registry("method")
        registry.register("grid", lambda: object())
        with self.assertRaisesRegex(KeyError, "grid"):
            registry.get("custom")


class SeedingTests(unittest.TestCase):
    def test_python_seed_is_reproducible_without_importing_tensorflow(self) -> None:
        first_report = seed_everything(73, include_tensorflow=False)
        first = [random.random() for _ in range(3)]
        second_report = seed_everything(73, include_tensorflow=False)
        second = [random.random() for _ in range(3)]
        self.assertEqual(first, second)
        self.assertEqual("73", os.environ["PYTHONHASHSEED"])
        self.assertTrue(first_report.python)
        self.assertEqual(first_report.numpy, second_report.numpy)


if __name__ == "__main__":
    unittest.main()
