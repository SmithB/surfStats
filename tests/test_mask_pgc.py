import numpy as np
import pytest

import surfStats.relief_map as relief_map
import surfStats.scale_map_fft2 as scale_map_fft2

# scale_map_fft2.mask_pgc and relief_map.mask_pgc are duplicate copies of the
# same function; test both so a fix applied to only one doesn't go unnoticed.
MASK_PGC_IMPLS = [scale_map_fft2.mask_pgc, relief_map.mask_pgc]


class FakeSub:
    """Minimal stand-in for the im_subset objects mask_pgc expects."""

    def __init__(self, z):
        self.z = z

    def setBounds(self, *bounds, update=False):
        pass


def _bitmask_matchtag(bad_bitmask_px=(2, 2), bad_matchtag_px=(3, 3), shape=(1, 6, 6)):
    bitmask = np.zeros(shape)
    bitmask[(0,) + bad_bitmask_px] = 1  # neither 0 nor 2 -> invalid
    matchtag = np.ones(shape, dtype=bool)
    matchtag[(0,) + bad_matchtag_px] = False
    return bitmask, matchtag


@pytest.mark.parametrize("mask_pgc", MASK_PGC_IMPLS)
class TestMaskPgc:
    def test_dec_le_1_ands_matchtag_directly(self, mask_pgc):
        bitmask, matchtag = _bitmask_matchtag()
        mask = np.ones((1, 6, 6), dtype=bool)
        pgc_subs = {"bitmask": FakeSub(bitmask), "matchtag": FakeSub(matchtag)}

        mask_pgc([0, 0, 6, 6], mask, pgc_subs, dec=1)

        expected = np.ones((1, 6, 6), dtype=bool)
        expected[0, 2, 2] = False
        expected[0, 3, 3] = False
        assert np.array_equal(mask, expected)

    def test_dec_le_1_accepts_a_uint8_matchtag(self, mask_pgc):
        # GDAL hands back a Byte band as uint8; `mask &= uint8_array` raises
        # UFuncTypeError because the result cannot be cast back to bool
        bitmask, matchtag = _bitmask_matchtag()
        pgc_subs = {
            "bitmask": FakeSub(bitmask),
            "matchtag": FakeSub(matchtag.astype(np.uint8)),
        }
        mask = np.ones((1, 6, 6), dtype=bool)

        mask_pgc([0, 0, 6, 6], mask, pgc_subs, dec=1)

        expected = np.ones((1, 6, 6), dtype=bool)
        expected[0, 2, 2] = False
        expected[0, 3, 3] = False
        assert mask.dtype == bool
        assert np.array_equal(mask, expected)

    def test_dec_gt_1_skips_erosion_when_matchtag_all_valid(self, mask_pgc):
        bitmask, _ = _bitmask_matchtag()
        matchtag_all_valid = np.ones((1, 6, 6), dtype=bool)
        mask = np.ones((1, 6, 6), dtype=bool)
        pgc_subs = {"bitmask": FakeSub(bitmask), "matchtag": FakeSub(matchtag_all_valid)}

        mask_pgc([0, 0, 6, 6], mask, pgc_subs, dec=4)

        # with no bad matchtag pixels, erosion is skipped entirely -- only
        # the bitmask-flagged pixel should be invalidated
        expected = np.ones((1, 6, 6), dtype=bool)
        expected[0, 2, 2] = False
        assert np.array_equal(mask, expected)

    def test_dec_gt_1_erodes_around_bad_matchtag_pixels(self, mask_pgc):
        bitmask, matchtag = _bitmask_matchtag()
        mask = np.ones((1, 6, 6), dtype=bool)
        pgc_subs = {"bitmask": FakeSub(bitmask), "matchtag": FakeSub(matchtag)}

        mask_pgc([0, 0, 6, 6], mask, pgc_subs, dec=4)

        # erosion around the bad matchtag pixel invalidates more than the
        # single bitmask-flagged pixel alone would
        assert mask.sum() < (6 * 6 - 1)
        assert mask[0, 2, 2] == False

    def test_updates_pgc_subs_bounds_before_masking(self, mask_pgc):
        bitmask, matchtag = _bitmask_matchtag()
        calls = []

        class RecordingSub(FakeSub):
            def setBounds(self, *bounds, update=False):
                calls.append(bounds)

        pgc_subs = {
            "bitmask": RecordingSub(bitmask),
            "matchtag": RecordingSub(matchtag),
        }
        mask = np.ones((1, 6, 6), dtype=bool)

        mask_pgc([1, 2, 6, 6], mask, pgc_subs, dec=1)

        assert calls == [(1, 2, 6, 6), (1, 2, 6, 6)]
