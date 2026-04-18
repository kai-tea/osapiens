import unittest

import numpy as np

from label_pipeline.constants import REPROJECT_NODATA
from label_pipeline.decoders import decode_gladl, decode_glads2, decode_radd


class DecoderTests(unittest.TestCase):
    def test_decode_radd_examples_and_filtering(self) -> None:
        raw = np.array([[20001, 30055, 21847, 0, REPROJECT_NODATA]], dtype=np.float32)
        obs, alert, conf, days = decode_radd(raw)

        np.testing.assert_array_equal(obs, np.array([[1, 1, 1, 1, 0]], dtype=np.uint8))
        np.testing.assert_array_equal(alert, np.array([[0, 0, 1, 0, 0]], dtype=np.uint8))
        np.testing.assert_allclose(conf, np.array([[0.0, 0.0, 0.5, 0.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(days, np.array([[-1, -1, 20, -1, -1]], dtype=np.int32))

    def test_decode_gladl_keeps_earliest_post_2020_alert(self) -> None:
        alert_2020 = np.array([[2, 0]], dtype=np.float32)
        date_2020 = np.array([[200, 0]], dtype=np.float32)
        alert_2021 = np.array([[3, 3]], dtype=np.float32)
        date_2021 = np.array([[10, 5]], dtype=np.float32)

        obs, alert, conf, days = decode_gladl(
            {20: alert_2020, 21: alert_2021},
            {20: date_2020, 21: date_2021},
        )

        np.testing.assert_array_equal(obs, np.array([[1, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(alert, np.array([[1, 1]], dtype=np.uint8))
        np.testing.assert_allclose(conf, np.array([[0.5, 1.0]], dtype=np.float32))
        np.testing.assert_array_equal(days, np.array([[199, 370]], dtype=np.int32))

    def test_decode_glads2_class_one_is_soft_only(self) -> None:
        alert_array = np.array([[1, 4, 0]], dtype=np.float32)
        date_array = np.array([[370, 400, REPROJECT_NODATA]], dtype=np.float32)

        obs, alert, conf, days = decode_glads2(alert_array, date_array)

        np.testing.assert_array_equal(obs, np.array([[1, 1, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(alert, np.array([[1, 1, 0]], dtype=np.uint8))
        np.testing.assert_allclose(conf, np.array([[0.25, 1.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(days, np.array([[5, 35, -1]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
