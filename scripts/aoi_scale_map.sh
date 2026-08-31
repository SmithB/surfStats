#!/usr/bin/env bash
#
# Clip+resample a GeoTIFF to an AOI polygon, then run scale_map_fft2.py on it.
#
# Usage:
#   aoi_scale_map.sh <geotiff> <aoi_geojson> <resolution> <n_lw> <output_dir> [--resampling MODE]

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <geotiff> <aoi_geojson> <resolution> <n_lw> <output_dir> [--resampling MODE]

  geotiff       Source GeoTIFF (any GDAL-readable raster).
  aoi_geojson   GeoJSON file containing the AOI polygon (used as a cutline).
  resolution    Target pixel size, in the source raster's CRS units.
  n_lw          N_LW argument forwarded to scale_map_fft2.py.
  output_dir    Directory for the generated VRT and scale_map_fft2.py outputs.

Options:
  --resampling MODE   gdalwarp resampling algorithm (default: bilinear).
  --tap               Snap the output grid to whole multiples of <resolution>
                      (gdalwarp -tap).  Required if the source was terrain-corrected
                      with SNAP's alignToStandardGrid, so the two grids coincide and
                      the clip becomes an exact crop instead of an interpolation.
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit "$([[ $# -lt 1 ]] && echo 1 || echo 0)"
fi

if [[ $# -lt 5 ]]; then
    echo "error: expected 5 positional arguments, got $#" >&2
    usage >&2
    exit 1
fi

geotiff=$1
aoi_geojson=$2
resolution=$3
n_lw=$4
output_dir=$5
shift 5

resampling=bilinear
tap=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resampling)
            if [[ $# -lt 2 ]]; then
                echo "error: --resampling requires an argument" >&2
                exit 1
            fi
            resampling=$2
            shift 2
            ;;
        --tap)
            tap=-tap
            shift
            ;;
        *)
            echo "error: unrecognized argument '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$geotiff" ]]; then
    echo "error: geotiff '$geotiff' does not exist or is not a file" >&2
    exit 1
fi

if [[ ! -f "$aoi_geojson" ]]; then
    echo "error: aoi_geojson '$aoi_geojson' does not exist or is not a file" >&2
    exit 1
fi

mkdir -p "$output_dir"

geotiff_stem=$(basename "${geotiff%.*}")
aoi_stem=$(basename "${aoi_geojson%.*}")
vrt_path="${output_dir%/}/${geotiff_stem}_${aoi_stem}_r${resolution}.vrt"

echo "building AOI-clipped VRT: $vrt_path"
gdalwarp -of VRT -ot Float32 \
    -tr "$resolution" "$resolution" \
    -r "$resampling" $tap \
    -cutline "$aoi_geojson" -crop_to_cutline \
    -dstnodata nan \
    "$geotiff" "$vrt_path"

echo "running scale_map_fft2.py on $vrt_path with N_LW=$n_lw"
scale_map_fft2.py "$vrt_path" "$n_lw"

echo "done."
echo "  vrt:        $vrt_path"
echo "  output dir: $output_dir"
