import tempfile
import unittest
from pathlib import Path

import numpy as np

from label_pipeline.build import _resolve_label_paths, _validate_label_inventory, synthesize_labels


class LabelSynthesisTests(unittest.TestCase):
    def test_label_synthesis_rules(self) -> None:
        radd_obs = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        gladl_obs = np.array([[1, 1, 1, 0]], dtype=np.uint8)
        glads2_obs = np.array([[1, 1, 1, 0]], dtype=np.uint8)

        radd_alert = np.array([[1, 0, 0, 0]], dtype=np.uint8)
        gladl_alert = np.array([[1, 0, 0, 0]], dtype=np.uint8)
        glads2_alert = np.array([[0, 0, 1, 0]], dtype=np.uint8)

        radd_conf = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        gladl_conf = np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
        glads2_conf = np.array([[0.0, 0.0, 0.25, 0.0]], dtype=np.float32)

        result = synthesize_labels(
            radd_obs=radd_obs,
            radd_alert=radd_alert,
            radd_conf=radd_conf,
            gladl_obs=gladl_obs,
            gladl_alert=gladl_alert,
            gladl_conf=gladl_conf,
            glads2_obs=glads2_obs,
            glads2_alert=glads2_alert,
            glads2_conf=glads2_conf,
        )

        np.testing.assert_array_equal(result["train_mask"], np.array([[1, 1, 1, 0]], dtype=np.uint8))
        np.testing.assert_array_equal(result["hard_label"], np.array([[1, 0, 255, 255]], dtype=np.uint8))
        np.testing.assert_array_equal(result["seed_mask"], np.array([[1, 1, 0, 0]], dtype=np.uint8))
        np.testing.assert_allclose(
            result["soft_target"],
            np.array([[0.5, 0.0, 0.08333334, -1.0]], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result["sample_weight"],
            np.array([[0.6666667, 1.0, 0.6666667, -1.0]], dtype=np.float32),
            atol=1e-6,
        )

        positive_mask = result["hard_label"] == 1
        negative_mask = result["hard_label"] == 0
        self.assertTrue(np.all(result["obs_count"][negative_mask] == 3))
        self.assertTrue(np.all(np.isin(result["hard_label"][result["seed_mask"] == 1], [0, 1])))
        self.assertTrue(np.all((result["soft_target"][result["train_mask"] == 1] >= 0.0)))
        self.assertTrue(np.all((result["soft_target"][result["train_mask"] == 1] <= 1.0)))
        self.assertTrue(np.all(positive_mask == np.array([[True, False, False, False]])))

    def test_missing_glads2_source_is_allowed_when_pair_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            labels_root = root / "labels" / "train"
            (labels_root / "radd").mkdir(parents=True)
            (labels_root / "gladl").mkdir(parents=True)
            (labels_root / "glads2").mkdir(parents=True)

            tile_id = "18NWG_6_6"
            (labels_root / "radd" / f"radd_{tile_id}_labels.tif").write_bytes(b"")
            (labels_root / "gladl" / f"gladl_{tile_id}_alert20.tif").write_bytes(b"")
            (labels_root / "gladl" / f"gladl_{tile_id}_alertDate20.tif").write_bytes(b"")

            resolved = _resolve_label_paths(root, tile_id)

            self.assertIsNone(resolved["glads2"])
            self.assertEqual(sorted(resolved["gladl"]), [20])

    def test_incomplete_glads2_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            labels_root = root / "labels" / "train"
            (labels_root / "radd").mkdir(parents=True)
            (labels_root / "gladl").mkdir(parents=True)
            (labels_root / "glads2").mkdir(parents=True)

            tile_id = "18NWG_6_6"
            (labels_root / "radd" / f"radd_{tile_id}_labels.tif").write_bytes(b"")
            (labels_root / "glads2" / f"glads2_{tile_id}_alert.tif").write_bytes(b"")

            with self.assertRaises(FileNotFoundError):
                _resolve_label_paths(root, tile_id)

    def test_validate_label_inventory_rejects_missing_source_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            labels_root = root / "labels" / "train"
            (labels_root / "radd").mkdir(parents=True)
            (labels_root / "gladl").mkdir(parents=True)
            (labels_root / "radd" / "dummy.tif").write_bytes(b"")
            (labels_root / "gladl" / "dummy.tif").write_bytes(b"")

            with self.assertRaises(FileNotFoundError):
                _validate_label_inventory(root)


if __name__ == "__main__":
    unittest.main()
