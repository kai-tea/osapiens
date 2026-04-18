import json
import tempfile
import unittest
from pathlib import Path

from label_pipeline.splits import load_split_assignments


class SplitLoadingTests(unittest.TestCase):
    def test_fold_assignments_region_fallback_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "splits" / "split_v1"
            metadata_dir = root / "metadata"
            split_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)

            (split_dir / "fold_assignments.csv").write_text("tile_id,fold_id\n18NWG_6_6,0\n18NWG_7_6,1\n")
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"tile_id": "18NWG_6_6", "country": "BR"},
                        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"tile_id": "18NWG_7_6", "country": "PE"},
                        "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                    },
                ],
            }
            (metadata_dir / "train_tiles.geojson").write_text(json.dumps(geojson))

            assignments = load_split_assignments(
                split_dir=split_dir,
                metadata_path=metadata_dir / "train_tiles.geojson",
            )

            self.assertEqual(assignments.loc[0, "region_id"], "BR")
            self.assertEqual(assignments.loc[1, "region_id"], "PE")

    def test_duplicate_tile_assignment_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "splits" / "split_v1"
            split_dir.mkdir(parents=True)
            (split_dir / "fold_assignments.csv").write_text(
                "tile_id,fold_id,region_id\n18NWG_6_6,0,BR\n18NWG_6_6,1,BR\n"
            )

            with self.assertRaises(ValueError):
                load_split_assignments(
                    split_dir=split_dir,
                    metadata_path=root / "metadata" / "train_tiles.geojson",
                )

    def test_val_tiles_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "splits" / "split_v1"
            split_dir.mkdir(parents=True)
            (split_dir / "val_tiles_fold0.csv").write_text("tile_id\n18NWG_6_6\n")
            (split_dir / "val_tiles_fold1.csv").write_text("tile_id\n18NWG_7_6\n")

            assignments = load_split_assignments(
                split_dir=split_dir,
                metadata_path=root / "metadata" / "train_tiles.geojson",
            )

            self.assertEqual(set(assignments["fold_id"]), {0, 1})
            self.assertEqual(assignments.loc[assignments["tile_id"] == "18NWG_6_6", "region_id"].item(), "18NWG")


if __name__ == "__main__":
    unittest.main()
