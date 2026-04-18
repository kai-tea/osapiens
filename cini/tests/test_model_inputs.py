import tempfile
import unittest
from pathlib import Path

import pandas as pd

from label_pipeline.model_inputs import (
    ParsedRemoteFile,
    build_file_manifest,
    build_tile_manifest,
    expected_local_path,
    filter_records_to_tiles,
    is_local_complete,
    parse_aef_filename,
    parse_remote_key,
    parse_s1_filename,
    parse_s2_filename,
)


class ModelInputsTests(unittest.TestCase):
    def test_filename_parsing(self) -> None:
        self.assertEqual(
            parse_s1_filename("18NWG_6_6__s1_rtc_2020_10_ascending.tif"),
            ("18NWG_6_6", 2020, 10, "ascending"),
        )
        self.assertEqual(
            parse_s2_filename("18NWG_6_6__s2_l2a_2020_1.tif"),
            ("18NWG_6_6", 2020, 1),
        )
        self.assertEqual(parse_aef_filename("18NWG_6_6_2025.tiff"), ("18NWG_6_6", 2025))

        s1 = parse_remote_key(
            "makeathon-challenge/sentinel-1/train/18NWG_6_6__s1_rtc/18NWG_6_6__s1_rtc_2020_10_descending.tif",
            remote_size=12,
        )
        self.assertEqual(s1.modality, "s1")
        self.assertEqual(s1.split, "train")
        self.assertEqual(s1.orbit, "descending")

        nested_s1 = parse_remote_key(
            "makeathon-challenge/sentinel-1/train/48PXC_7_7__s1_rtc/48PXC_7_7__s1_rtc_2020_10_ascending/96c6ee62-fc9e-48fe-b135-f70ffa804058.tif",
            remote_size=12,
        )
        self.assertEqual(nested_s1.tile_id, "48PXC_7_7")
        self.assertEqual(nested_s1.year, 2020)
        self.assertEqual(nested_s1.month, 10)
        self.assertEqual(nested_s1.orbit, "ascending")

        with self.assertRaises(ValueError):
            parse_remote_key("makeathon-challenge/sentinel-2/train/18NWG_6_6__s2_l2a/.DS_Store")

    def test_manifest_generation_and_completeness(self) -> None:
        repo_root = Path.cwd()
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp_dir:
            tmp = Path(tmp_dir)
            data_root = tmp / "data" / "makeathon-challenge"
            labelpack_path = tmp / "artifacts" / "labels_v1" / "tiles" / "18NWG_6_6_labelpack.tif"
            labelpack_path.parent.mkdir(parents=True)
            labelpack_path.write_bytes(b"label")

            records = [
                ParsedRemoteFile(
                    "makeathon-challenge/sentinel-1/train/18NWG_6_6__s1_rtc/18NWG_6_6__s1_rtc_2020_1_descending.tif",
                    "train",
                    "18NWG_6_6",
                    "s1",
                    2020,
                    1,
                    "descending",
                    3,
                ),
                ParsedRemoteFile(
                    "makeathon-challenge/sentinel-1/train/18NWG_6_6__s1_rtc/18NWG_6_6__s1_rtc_2020_1_ascending.tif",
                    "train",
                    "18NWG_6_6",
                    "s1",
                    2020,
                    1,
                    "ascending",
                    3,
                ),
                ParsedRemoteFile(
                    "makeathon-challenge/sentinel-2/train/18NWG_6_6__s2_l2a/18NWG_6_6__s2_l2a_2020_1.tif",
                    "train",
                    "18NWG_6_6",
                    "s2",
                    2020,
                    1,
                    None,
                    3,
                ),
                ParsedRemoteFile(
                    "makeathon-challenge/aef-embeddings/train/18NWG_6_6_2020.tiff",
                    "train",
                    "18NWG_6_6",
                    "aef",
                    2020,
                    None,
                    None,
                    3,
                ),
                ParsedRemoteFile(
                    "makeathon-challenge/sentinel-2/test/18NVJ_1_6__s2_l2a/18NVJ_1_6__s2_l2a_2020_1.tif",
                    "test",
                    "18NVJ_1_6",
                    "s2",
                    2020,
                    1,
                    None,
                    3,
                ),
            ]

            for record in records:
                path = expected_local_path(record.remote_key, data_root)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"abc")

            tile_table = pd.DataFrame(
                [
                    {"split": "train", "tile_id": "18NWG_6_6", "fold_id": 0, "region_id": "18NWG"},
                    {"split": "test", "tile_id": "18NVJ_1_6", "fold_id": pd.NA, "region_id": "18NVJ"},
                ]
            )
            label_manifest = pd.DataFrame(
                [
                    {
                        "tile_id": "18NWG_6_6",
                        "labelpack_path": labelpack_path.relative_to(repo_root).as_posix(),
                        "s2_ref_path": expected_local_path(records[2].remote_key, data_root)
                        .relative_to(repo_root)
                        .as_posix(),
                    }
                ]
            )

            filtered = filter_records_to_tiles(records, tile_table)
            file_manifest = build_file_manifest(
                filtered,
                tile_table=tile_table,
                data_root=data_root,
                label_manifest=label_manifest,
            )
            tile_manifest = build_tile_manifest(
                tile_table=tile_table,
                file_manifest=file_manifest,
                data_root=data_root,
                label_manifest=label_manifest,
            )

            s1_rows = file_manifest[file_manifest["modality"] == "s1"].sort_values("sequence_order")
            self.assertEqual(s1_rows.iloc[0]["orbit"], "ascending")
            self.assertEqual(s1_rows.iloc[1]["orbit"], "descending")
            self.assertEqual(int(file_manifest["exists_local"].sum()), len(records))
            self.assertTrue(all(is_local_complete(record, data_root) for record in records))

            train_tile = tile_manifest[tile_manifest["tile_id"] == "18NWG_6_6"].iloc[0]
            self.assertEqual(train_tile["label_available"], 1)
            self.assertEqual(train_tile["all_modalities_complete"], 1)

            test_tile = tile_manifest[tile_manifest["tile_id"] == "18NVJ_1_6"].iloc[0]
            self.assertEqual(test_tile["labelpack_path"], "")
            self.assertEqual(test_tile["s2_complete"], 1)
            self.assertEqual(test_tile["s1_complete"], 0)
            self.assertEqual(test_tile["all_modalities_complete"], 0)


if __name__ == "__main__":
    unittest.main()
