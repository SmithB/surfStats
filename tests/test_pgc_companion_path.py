import pytest

from surfStats.scale_map_fft2 import pgc_companion_path


@pytest.mark.parametrize(
    "input_file, suffix, expected",
    [
        ("/data/foo_dem.tif", "matchtag", "/data/foo_matchtag.tif"),
        ("/data/foo_dem.tif", "bitmask", "/data/foo_bitmask.tif"),
        ("/data/foo_dem.vrt", "matchtag", "/data/foo_matchtag.tif"),
        ("/data/foo_dem.vrt", "bitmask", "/data/foo_bitmask.tif"),
        ("foo_dem.tif", "matchtag", "foo_matchtag.tif"),  # relative path
        ("/data/foo.tif", "matchtag", "/data/foo_matchtag.tif"),  # no "_dem" stem
    ],
)
def test_pgc_companion_path(input_file, suffix, expected):
    assert pgc_companion_path(input_file, suffix) == expected
