"""
in1AutoATES.py
Input handling and preprocessing for autoATES v3.
Simplified: Only DEM + Forest Canopy + optional AOI.
"""

from pathlib import Path
import logging
from osgeo import gdal, ogr, osr
import numpy as np
import rasterio

gdal.UseExceptions()

logger = logging.getLogger(__name__)


def align_and_clip_rasters(
    dem_path: Path,
    forest_canopy_path: Path,
    output_dir: Path,
    aoi_path: Path | None = None,
    target_resolution: tuple[float, float] | None = None,
    output_basename: str = "autoATES"
) -> list[tuple[Path, Path]]:
    """
    Aligns DEM and forest canopy raster to common resolution and extent.
    Uses DEM as reference grid.
    """
    logger.info("Starting raster alignment and preprocessing...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open reference DEM
    ref_ds = gdal.Open(str(dem_path))
    if ref_ds is None:
        raise FileNotFoundError(f"Cannot open DEM: {dem_path}")

    ref_transform = ref_ds.GetGeoTransform()
    ref_projection = ref_ds.GetProjection()

    x_res = target_resolution[0] if target_resolution else abs(ref_transform[1])
    y_res = target_resolution[1] if target_resolution else abs(ref_transform[5])

    nodata = -9999

    logger.info(f"Target resolution: {x_res:.2f} x {y_res:.2f} m")

    if aoi_path and aoi_path.exists():
        logger.info(f"Clipping to AOI: {aoi_path}")
        outputs = _warp_with_aoi(dem_path, forest_canopy_path, output_dir, 
                               output_basename, aoi_path, x_res, y_res, ref_projection, nodata)
    else:
        logger.info("No AOI provided → processing full extent")
        outputs = _warp_full_extent(dem_path, forest_canopy_path, output_dir, 
                                  output_basename, x_res, y_res, ref_projection, nodata)

    logger.info("Raster preprocessing completed successfully.")
    return outputs


def _warp_with_aoi(dem_path, canopy_path, output_dir, basename, aoi_path, x_res, y_res, ref_projection, nodata):
    """Process with AOI clipping."""
    outputs = []

    shp_ds = ogr.Open(str(aoi_path))
    layer = shp_ds.GetLayer()

    for i, feature in enumerate(layer):
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        temp_shp = output_dir / f"temp_feature_{i:02d}.shp"
        _create_single_feature_shapefile(geom, temp_shp, layer.GetSpatialRef())

        suffix = f"_{i:02d}"
        
        dsm_out = _warp_file(dem_path, f"{basename}_DSM{suffix}.tif", output_dir,
                           temp_shp, x_res, y_res, ref_projection, nodata, gdal.GRA_Bilinear)
        # Forest canopy: 0 % == open terrain (valid, and exactly where avalanches
        # start) - do NOT treat 0 as NoData, or open alpine release zones vanish.
        canopy_out = _warp_file(canopy_path, f"{basename}_FOREST_CANOPY{suffix}.tif", output_dir,
                              temp_shp, x_res, y_res, ref_projection, nodata, gdal.GRA_NearestNeighbour,
                              zero_to_nodata=False)

        outputs.append((dsm_out, canopy_out))

        # Cleanup
        for ext in [".shp", ".shx", ".dbf", ".prj"]:
            temp_shp.with_suffix(ext).unlink(missing_ok=True)

    return outputs


def _warp_full_extent(dem_path, canopy_path, output_dir, basename, x_res, y_res, ref_projection, nodata):
    """Process full extent (no AOI)."""
    dsm_out = _warp_file(dem_path, f"{basename}_DSM.tif", output_dir, None,
                        x_res, y_res, ref_projection, nodata, gdal.GRA_Bilinear)
    # Forest canopy: 0 % == open terrain (valid) - keep it, see _warp_with_aoi.
    canopy_out = _warp_file(canopy_path, f"{basename}_FOREST_CANOPY.tif", output_dir, None,
                           x_res, y_res, ref_projection, nodata, gdal.GRA_NearestNeighbour,
                           zero_to_nodata=False)
    
    return [(dsm_out, canopy_out)]


def _warp_file(src_path, out_name, output_dir, cutline_shp, x_res, y_res, ref_projection, nodata, resample_alg,
               zero_to_nodata=True):
    """Generic warp helper with strict NoData handling.

    zero_to_nodata: when True (DEM), pixels equal to 0 are forced to NoData
    (void-fill guard). Must be False for the forest canopy raster, where 0 %
    is valid open terrain - forcing it to NoData erases open release zones.
    """
    out_path = output_dir / out_name
    
    opts = {
        "format": "GTiff",
        "xRes": x_res,
        "yRes": y_res,
        "dstSRS": ref_projection,
        "dstNodata": nodata,           # Enforce NoData
        "multithread": True,
        "outputType": gdal.GDT_Float32,   # Use Float32 for better NoData support
        "resampleAlg": resample_alg,
    }
    
    if cutline_shp:
        opts.update({
            "cutlineDSName": str(cutline_shp),
            "cropToCutline": True,
        })

    gdal.Warp(str(out_path), str(src_path), **opts)
    
    # Extra safety: Force NoData value using rasterio
    with rasterio.open(out_path, 'r+') as src:
        data = src.read(1)
        if zero_to_nodata:
            data = np.where(np.isnan(data) | ((data == 0) & (src.nodata != 0)), nodata, data)
        else:
            # Forest canopy: only NaN is NoData; keep 0 % (open terrain).
            data = np.where(np.isnan(data), nodata, data)
        src.write(data, 1)
        src.nodata = nodata   # Ensure metadata is correct

    return out_path


def _create_single_feature_shapefile(geometry, output_path: Path, srs):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if output_path.exists():
        driver.DeleteDataSource(str(output_path))
    
    ds = driver.CreateDataSource(str(output_path))
    layer = ds.CreateLayer("feature", srs=srs, geom_type=ogr.wkbPolygon)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(geometry.Clone())
    layer.CreateFeature(feat)
    ds = None


# For quick testing
if __name__ == "__main__":
    pass