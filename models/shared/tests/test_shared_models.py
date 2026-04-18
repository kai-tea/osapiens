import tempfile
import unittest
from pathlib import Path

import pandas as pd

from models.shared.config import load_simple_config
from models.shared.contract import TEAM_MEMBERS, output_dir_for_owner, required_manifest_paths
from models.shared.data import load_manifests, split_tiles
from models.shared.evaluation import binary_metrics, validate_prediction_schema


class SharedModelTests(unittest.TestCase):
    def test_personal_configs_load(self) -> None:
        for owner in TEAM_MEMBERS:
            with self.subTest(owner=owner):
                config = load_simple_config(Path("models") / owner / "config.example.yaml")
                self.assertEqual(config["owner"], owner)
                self.assertEqual(config["active_fold"], 0)
                self.assertIn(owner, config["model_name"])

    def test_output_dirs_are_owner_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for owner in TEAM_MEMBERS:
                expected = Path(tmp_dir) / "artifacts" / "models" / owner
                self.assertEqual(output_dir_for_owner(owner, tmp_dir), expected)

    def test_split_tiles_returns_non_empty_sets(self) -> None:
        tile_manifest = pd.DataFrame(
            [
                {"split": "train", "tile_id": "tile_a", "fold_id": 0},
                {"split": "train", "tile_id": "tile_b", "fold_id": 1},
                {"split": "train", "tile_id": "tile_c", "fold_id": 2},
                {"split": "test", "tile_id": "tile_test", "fold_id": pd.NA},
            ]
        )
        folds = split_tiles(tile_manifest, active_fold=0)
        self.assertEqual(list(folds.val_tiles["tile_id"]), ["tile_a"])
        self.assertEqual(set(folds.train_tiles["tile_id"]), {"tile_b", "tile_c"})
        self.assertEqual(list(folds.test_tiles["tile_id"]), ["tile_test"])

    def test_local_artifact_contract_when_available(self) -> None:
        paths = required_manifest_paths()
        missing = [path for path in paths.values() if not path.exists()]
        if missing:
            self.skipTest("Generated artifacts are not available locally")

        manifests = load_manifests()
        folds = split_tiles(manifests.tile_manifest, active_fold=0)
        self.assertGreater(len(folds.train_tiles), 0)
        self.assertGreater(len(folds.val_tiles), 0)
        self.assertGreater(len(folds.test_tiles), 0)
        self.assertGreater(len(manifests.pixel_index), 0)

    def test_prediction_schema_and_metrics(self) -> None:
        predictions = pd.DataFrame(
            {
                "tile_id": ["a", "a", "b"],
                "row": [0, 1, 2],
                "col": [0, 1, 2],
                "y_true": [1, 0, 1],
                "score": [0.9, 0.2, 0.4],
                "fold_id": [0, 0, 0],
                "model_name": ["demo", "demo", "demo"],
            }
        )
        validate_prediction_schema(predictions, split="validation")
        metrics = binary_metrics(predictions["y_true"], predictions["score"], threshold=0.5)
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(metrics["fn"], 1.0)
        self.assertGreater(metrics["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
