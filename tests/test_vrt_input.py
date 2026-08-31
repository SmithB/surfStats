import numpy as np
import pytest
from osgeo import gdal

from surfStats import im_subset
from surfStats.scale_map_fft2 import _open_or_die


def _make_geotiff(path, data, nodata=-9999.0):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(path), data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(data)
    ds.FlushCache()
    ds = None
    return path


def test_im_subset_reads_vrt_and_geotiff_identically(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, 20)).astype(np.float32)
    tif_path = _make_geotiff(tmp_path / "test_dem.tif", data)
    vrt_path = tmp_path / "test_dem.vrt"
    gdal.BuildVRT(str(vrt_path), [str(tif_path)])

    tif_ds = gdal.Open(str(tif_path))
    vrt_ds = gdal.Open(str(vrt_path))

    tif_sub = im_subset(0, 0, 20, 20, tif_ds, Bands=[1])
    tif_sub.copySubsetFrom()
    vrt_sub = im_subset(0, 0, 20, 20, vrt_ds, Bands=[1])
    vrt_sub.copySubsetFrom()

    assert np.array_equal(tif_sub.z, vrt_sub.z)
    assert tif_ds.GetRasterBand(1).GetNoDataValue() == vrt_ds.GetRasterBand(1).GetNoDataValue()
    assert tif_ds.GetGeoTransform() == vrt_ds.GetGeoTransform()


def test_im_subset_reads_windowed_subset_identically(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(size=(50, 50)).astype(np.float32)
    tif_path = _make_geotiff(tmp_path / "win_dem.tif", data)
    vrt_path = tmp_path / "win_dem.vrt"
    gdal.BuildVRT(str(vrt_path), [str(tif_path)])

    tif_ds = gdal.Open(str(tif_path))
    vrt_ds = gdal.Open(str(vrt_path))

    tif_sub = im_subset(10, 10, 15, 15, tif_ds, Bands=[1])
    tif_sub.copySubsetFrom()
    vrt_sub = im_subset(10, 10, 15, 15, vrt_ds, Bands=[1])
    vrt_sub.copySubsetFrom()

    assert np.array_equal(tif_sub.z, vrt_sub.z)


def test_open_or_die_raises_clear_error_for_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="does_not_exist"):
        _open_or_die(str(tmp_path / "does_not_exist.tif"), "input file")
