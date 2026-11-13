import unittest
from unittest.mock import patch

import main


class ModelStartupTests(unittest.TestCase):
    def tearDown(self):
        main.model = None

    def test_startup_fails_if_model_download_fails(self):
        with patch.object(main.os.path, "isfile", return_value=False), patch.object(
            main, "hf_hub_download", side_effect=OSError("network unavailable")
        ):
            with self.assertRaises(RuntimeError):
                main.load_model()
        self.assertIsNone(main.model)

    def test_startup_loads_existing_model(self):
        sentinel = object()
        with patch.object(main.os.path, "isfile", return_value=True), patch.object(
            main, "YOLO", return_value=sentinel
        ):
            main.load_model()
        self.assertIs(main.model, sentinel)


class UsageLimitTests(unittest.TestCase):
    def setUp(self):
        main.usage_tracker.clear()
        self.original_limit = main.FREE_LIMIT
        main.FREE_LIMIT = 1

    def tearDown(self):
        main.FREE_LIMIT = self.original_limit
        main.usage_tracker.clear()

    def test_limit_blocks_after_allowed_request(self):
        main.check_limit("198.51.100.10")
        with self.assertRaises(main.HTTPException) as error:
            main.check_limit("198.51.100.10")
        self.assertEqual(error.exception.status_code, 429)

    def test_limit_is_separate_per_client(self):
        main.check_limit("198.51.100.10")
        main.check_limit("198.51.100.11")


if __name__ == "__main__":
    unittest.main()
