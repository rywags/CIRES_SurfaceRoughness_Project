import numpy as np
import glob
import re
import rasterio
import os
import pyproj
import pandas as pd
from datetime import datetime
import tarfile
import json
import fnmatch
import rasterio.windows


def main() -> None:
    LONGITUDES = np.arange(start=-85.0, stop=-110.0, step=-0.5)
    LATITUDES = np.arange(start=-74.5, stop=-79.5, step=-0.25)

    LONGITUDES_GRID, LATITUDES_GRID = np.meshgrid(LONGITUDES, LATITUDES)
    WGS84_PROJECTION = pyproj.CRS("EPSG:4326")

    BASE_DIRECTORY = r"E:\\Surface Roughness Processing"
    DATA_DIRECTORIES = glob.glob(os.path.join(BASE_DIRECTORY, "*"))

    PATHROW_PATTERN = re.compile(r".*P(\d{3})R(\d{3})_(\d{7})$")
    PATHROW_DIRECTORIES = list(filter(PATHROW_PATTERN.fullmatch, DATA_DIRECTORIES))

    with rasterio.open(
        r"E:\\Surface Roughness Processing\\REMA_elev+slope\\rema_62.5m_mosaic.tif"
    ) as REMA_ELEVATION, rasterio.open(
        r"E:\\Surface Roughness Processing\\REMA_elev+slope\\rema_62.5m_mosaic_slope.tif"
    ) as REMA_MAGNITUDE, rasterio.open(
        r"E:\\Surface Roughness Processing\\REMA_elev+slope\\rema_62.5m_mosaic_slope_aspect.tif"
    ) as REMA_DIRECTION:
        ANTARCTIC_PROJECTION = pyproj.CRS(REMA_ELEVATION.crs)
        ANTARCTIC_TO_WGS84 = pyproj.Transformer.from_crs(
            ANTARCTIC_PROJECTION, WGS84_PROJECTION, always_xy=True
        )

        ANTARCTIC_TRANSFORMER = pyproj.Transformer.from_crs(
            WGS84_PROJECTION, ANTARCTIC_PROJECTION, always_xy=True
        )
        LONG_COORDINATES, LAT_COORDINATES = ANTARCTIC_TRANSFORMER.transform(
            LONGITUDES_GRID, LATITUDES_GRID
        )

        LONGITUDE_1D = LONG_COORDINATES.flatten()
        LATITUDE_1D = LAT_COORDINATES.flatten()

        for directory in PATHROW_DIRECTORIES:
            path, row, yyyydoy_date = PATHROW_PATTERN.match(directory).groups()
            yyyymmdd_date = datetime.strptime(yyyydoy_date, "%Y%j").strftime("%Y%m%d")

            filtered_image = os.path.join(directory, f"filtered_image_{yyyydoy_date}.TIF")
            correlation_image = os.path.join(directory, f"correlation_image_{yyyydoy_date}.TIF")

            landsat_tarpath = f"E:\\Surface_Roughness_Data\\p{path}_r{row}\\earthexplorer\\LC08_L1GT_{path}{row}_{yyyymmdd_date}_*_02_T2.tar"
            landsat_tarfile = glob.glob(landsat_tarpath)[0]

            with tarfile.open(landsat_tarfile, "r") as landsat_tar:
                tar_members = landsat_tar.getnames()
                json_filename = (
                    f"LC08_L1GT_{path}{row}_{yyyymmdd_date}_*_02_T2_MTL.json"
                )
                json_mtl = [
                    member
                    for member in tar_members
                    if fnmatch.fnmatch(member, json_filename)
                ][0]

                with landsat_tar.extractfile(json_mtl) as mtl:
                    metadata = json.load(mtl)
                    sun_elevation = metadata["LANDSAT_METADATA_FILE"][
                        "IMAGE_ATTRIBUTES"
                    ]["SUN_ELEVATION"]

            data_list = []  # Store data for this directory

            with rasterio.open(filtered_image) as filtered, rasterio.open(
                correlation_image
            ) as correlation:
                filtered_bounds = filtered.bounds
                correlation_bounds = correlation.bounds

                for lon, lat in zip(LONGITUDE_1D, LATITUDE_1D):
                    if (
                        filtered_bounds.left <= lon <= filtered_bounds.right
                        and filtered_bounds.bottom <= lat <= filtered_bounds.top
                        and correlation_bounds.left <= lon <= correlation_bounds.right
                        and correlation_bounds.bottom <= lat <= correlation_bounds.top
                    ):
                        rema_long, rema_lat = REMA_ELEVATION.index(lon, lat)
                        rema_window = rasterio.windows.Window(rema_long, rema_lat, 1, 1)
                        rema_value = REMA_ELEVATION.read(1, window=rema_window)[0][0]

                        rema_long2, rema_lat2 = REMA_MAGNITUDE.index(lon, lat)
                        rema_window2 = rasterio.windows.Window(rema_long2, rema_lat2, 1, 1)
                        rema_value2 = REMA_MAGNITUDE.read(1, window=rema_window2)[0][0]

                        rema_long3, rema_lat3 = REMA_DIRECTION.index(lon, lat)
                        rema_window3 = rasterio.windows.Window(rema_long3, rema_lat3, 1, 1)
                        rema_value3 = REMA_DIRECTION.read(1, window=rema_window3)[0][0]

                        filtered_long, filtered_lat = filtered.index(lon, lat)
                        filtered_window = rasterio.windows.Window(filtered_long, filtered_lat, 1, 1)
                        filtered_read = filtered.read(1, window=filtered_window)
                        filtered_value = filtered_read[0][0] if filtered_read.size > 0 else np.nan

                        correlation_long, correlation_lat = correlation.index(lon, lat)
                        correlation_window = rasterio.windows.Window(correlation_long, correlation_lat, 1, 1)
                        correlation_read = correlation.read(1, window=correlation_window)
                        correlation_value = correlation_read[0][0] if correlation_read.size > 0 else np.nan

                        # **Transform back to global coordinates**
                        wgs84_lon, wgs84_lat = ANTARCTIC_TO_WGS84.transform(lon, lat)

                        # **Round Longitude and Latitude to 2 decimal places**
                        wgs84_lon = round(wgs84_lon, 2)
                        wgs84_lat = round(wgs84_lat, 2)

                        data_list.append({
                            "Longitude (WGS84)": wgs84_lon,
                            "Latitude (WGS84)": wgs84_lat,
                            "Filtered Value": filtered_value,
                            "Correlation Value": correlation_value,
                            "Solar Elevation": sun_elevation,
                            "REMA Elevation": rema_value,
                            "REMA Slope Magnitude": rema_value2,
                            "REMA Slope Direction": rema_value3
                        })

            # Save data to a CSV file in the corresponding directory
            output_filename = os.path.join(directory, f"P{path}R{row}_{yyyydoy_date}.csv")
            df = pd.DataFrame(data_list)
            df.to_csv(output_filename, index=False)
            print(f"Data saved to {output_filename}")


if __name__ == "__main__":
    main()
