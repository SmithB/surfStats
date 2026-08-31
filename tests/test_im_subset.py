import numpy as np
import pytest

from osgeo import gdal

from surfStats.im_subset import gdal_dtype, im_subset, match_range


def _mem_ds(array, gdal_type):
    """A single-band in-memory GDAL dataset holding `array`."""
    ny, nx = array.shape
    ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal_type)
    ds.GetRasterBand(1).WriteArray(array)
    return ds


class TestMatchRange:
    def test_identical_ranges_overlap_completely(self):
        assert match_range(0, 10, 0, 10) == (0, 10, 0, 10, True)

    def test_destination_offset_inside_source(self):
        si0, si1, di0, di1, valid = match_range(0, 10, 3, 4)
        assert (si0, si1, di0, di1) == (3, 7, 0, 4)
        assert valid

    def test_destination_hanging_off_the_start(self):
        si0, si1, di0, di1, valid = match_range(0, 10, -2, 5)
        assert (si0, si1, di0, di1) == (0, 3, 2, 5)
        assert valid

    def test_destination_hanging_off_the_end(self):
        si0, si1, di0, di1, valid = match_range(0, 10, 8, 5)
        assert (si0, si1, di0, di1) == (8, 10, 0, 2)
        assert valid

    def test_disjoint_ranges_are_not_valid(self):
        assert match_range(0, 10, 20, 5)[4] == False


class TestGdalDtype:
    @pytest.mark.parametrize(
        "type_name, expected",
        [
            ("Byte", np.ubyte),
            ("Int16", np.int16),
            ("UInt16", np.uint16),
            ("Int32", np.int32),
            ("UInt32", np.uint32),
            # Int64 used to be mapped to np.int32, which silently truncated
            ("Int64", np.int64),
            ("UInt64", np.uint64),
            ("Float32", np.float32),
            ("Float64", np.float64),
            # GDAL >= 3.7 reports this type name; it used to raise KeyError
            ("Int8", np.int8),
        ],
    )
    def test_known_types(self, type_name, expected):
        assert gdal_dtype(type_name) is expected
        assert gdal_dtype(type_name.lower()) is expected

    def test_unknown_type_names_the_offending_type(self):
        with pytest.raises(KeyError, match="CFloat32"):
            gdal_dtype("CFloat32")


class TestCopySubsetFromDtype:
    def test_int64_raster_is_not_truncated(self):
        # a value that does not survive a round trip through int32
        big = np.int64(2) ** 40 + 12345
        array = np.full((4, 4), big, dtype=np.int64)
        sub = im_subset(0, 0, 4, 4, _mem_ds(array, gdal.GDT_Int64), Bands=[1])
        sub.copySubsetFrom()
        assert sub.z.dtype == np.int64
        assert np.all(sub.z[0] == big)

    @pytest.mark.skipif(
        not hasattr(gdal, "GDT_Int8"), reason="GDAL < 3.7 has no Int8 type"
    )
    def test_int8_raster_reads_without_a_keyerror(self):
        array = np.array([[-128, -1], [0, 127]], dtype=np.int8)
        sub = im_subset(0, 0, 2, 2, _mem_ds(array, gdal.GDT_Int8), Bands=[1])
        sub.copySubsetFrom()
        assert sub.z.dtype == np.int8
        assert np.array_equal(sub.z[0], array)


class TestWriteSubsetTo:
    def test_write_into_an_in_memory_target_uses_zero_based_bands(self):
        ds = gdal.GetDriverByName("MEM").Create("", 6, 6, 2, gdal.GDT_Float32)
        for band in (1, 2):
            ds.GetRasterBand(band).WriteArray(np.zeros((6, 6), dtype=np.float32))

        base = im_subset(0, 0, 6, 6, ds, Bands=[1, 2])
        base.copySubsetFrom()
        assert base.level == 0

        # a level>0 target routes writeSubsetTo through the array-copy branch
        target = im_subset(0, 0, 6, 6, base, Bands=[1, 2])
        assert target.level > 0

        src = im_subset(0, 0, 6, 6, base, Bands=[1, 2])
        src.z = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6)

        # band numbers are 1-based; indexing z with them directly used to write
        # band 1 into row 1 and run off the end of a 2-band array for band 2
        src.writeSubsetTo([1, 2], target)

        assert np.array_equal(base.z, src.z)


def test_default_stride_steps_one_pixel():
    # stride=None used to build np.array([None, None]), which left self.stride
    # holding None for any later __getitem__ call
    sub = im_subset(0, 0, 4, 4, None, Bands=[1])
    assert np.array_equal(sub.stride, np.array([1, 1]))
    assert sub.xy0.shape == [4, 4]
