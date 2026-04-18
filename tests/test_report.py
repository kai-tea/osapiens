import unittest

import pandas as pd

from label_pipeline.report import (
    build_overview,
    build_region_summary,
    build_tile_summary,
    format_markdown_report,
)


class ReportTests(unittest.TestCase):
    def test_report_summary_flags_missing_glads2_and_no_hard_negatives(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "tile_id": "tile_a",
                    "fold_id": 0,
                    "region_id": "region_1",
                    "labelpack_path": "artifacts/labels_v1/tiles/tile_a_labelpack.tif",
                    "s2_ref_path": "data/makeathon-challenge/sentinel-2/train/tile_a__s2_l2a/tile_a_2020_1.tif",
                    "n_train_pixels": 3,
                    "n_hard_pos": 1,
                    "n_hard_neg": 0,
                    "n_seed_pos": 1,
                    "n_seed_neg": 0,
                    "radd_pos_rate": 0.2,
                    "gladl_pos_rate": 0.3,
                    "glads2_pos_rate": 0.1,
                },
                {
                    "tile_id": "tile_b",
                    "fold_id": 1,
                    "region_id": "region_2",
                    "labelpack_path": "artifacts/labels_v1/tiles/tile_b_labelpack.tif",
                    "s2_ref_path": "data/makeathon-challenge/sentinel-2/train/tile_b__s2_l2a/tile_b_2020_1.tif",
                    "n_train_pixels": 2,
                    "n_hard_pos": 1,
                    "n_hard_neg": 0,
                    "n_seed_pos": 1,
                    "n_seed_neg": 0,
                    "radd_pos_rate": 0.4,
                    "gladl_pos_rate": 0.5,
                    "glads2_pos_rate": 0.0,
                },
            ]
        )
        pixel_index = pd.DataFrame(
            [
                {"tile_id": "tile_a", "region_id": "region_1", "fold_id": 0, "obs_count": 3, "hard_label": 1, "seed_mask": 1},
                {"tile_id": "tile_a", "region_id": "region_1", "fold_id": 0, "obs_count": 2, "hard_label": 255, "seed_mask": 0},
                {"tile_id": "tile_a", "region_id": "region_1", "fold_id": 0, "obs_count": 2, "hard_label": 255, "seed_mask": 0},
                {"tile_id": "tile_b", "region_id": "region_2", "fold_id": 1, "obs_count": 2, "hard_label": 1, "seed_mask": 1},
                {"tile_id": "tile_b", "region_id": "region_2", "fold_id": 1, "obs_count": 2, "hard_label": 255, "seed_mask": 0},
            ]
        )
        tile_source_stats = pd.DataFrame(
            [
                {
                    "tile_id": "tile_a",
                    "radd_obs_pixels": 10,
                    "radd_available": 1,
                    "gladl_obs_pixels": 9,
                    "gladl_available": 1,
                    "glads2_obs_pixels": 8,
                    "glads2_available": 1,
                },
                {
                    "tile_id": "tile_b",
                    "radd_obs_pixels": 10,
                    "radd_available": 1,
                    "gladl_obs_pixels": 8,
                    "gladl_available": 1,
                    "glads2_obs_pixels": 0,
                    "glads2_available": 0,
                },
            ]
        )
        overlap = pd.DataFrame(
            [
                {
                    "scope_type": "overall",
                    "scope_value": "all",
                    "pair": "radd_gladl",
                    "n_obs_overlap": 100,
                    "n_pos_left": 10,
                    "n_pos_right": 12,
                    "n_pos_both": 6,
                    "disagreement_rate": 0.2,
                    "jaccard_pos": 0.375,
                }
            ]
        )

        tile_summary = build_tile_summary(manifest, pixel_index, tile_source_stats)
        region_summary = build_region_summary(tile_summary)
        overview = build_overview(tile_summary, region_summary, overlap)
        report = format_markdown_report(overview, tile_summary, region_summary)

        self.assertEqual(tile_summary.loc[tile_summary["tile_id"] == "tile_b", "missing_glads2"].item(), 1)
        self.assertEqual(region_summary.loc[region_summary["region_id"] == "region_2", "tiles_missing_glads2"].item(), 1)
        self.assertTrue(overview["no_hard_negatives"])
        self.assertEqual(overview["total_train_pixels"], 5)
        self.assertIn("v1 currently has no hard negatives", report)


if __name__ == "__main__":
    unittest.main()
