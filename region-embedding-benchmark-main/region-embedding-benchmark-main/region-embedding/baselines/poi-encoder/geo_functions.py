import pandas
import geopandas
import h3
import numpy as np
from shapely import wkt
from h3ronpy import cells_to_string, grid_disk
from h3ronpy.vector import ContainmentMode, cells_to_wkb_polygons, wkb_to_cells
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from collections.abc import Iterable

__all__ = [
    "haversine_np",
    "H3Interpolation",
]

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378137 * c
    return km
class H3Interpolation:
    def __init__(self, gdf) -> None:
        self.gdf = gdf

    def _shapely_geometry_to_h3(
        self,
        geometry,
        h3_resolution: int,
        buffer: bool = True,
    ) -> list[str]:
        if not (0 <= h3_resolution <= 15):
            raise ValueError(f"Resolution {h3_resolution} is not between 0 and 15.")

        # -> garanta lista concreta de WKB (nada de generator)
        if isinstance(geometry, geopandas.GeoSeries):
            wkb = list(geometry.to_wkb().to_numpy())
        elif isinstance(geometry, geopandas.GeoDataFrame):
            wkb = list(geometry["geometry"].to_wkb().to_numpy())
        elif isinstance(geometry, Iterable) and not isinstance(geometry, (str, bytes)):
            geoms = list(geometry)
            wkb = [g.wkb if isinstance(g, BaseGeometry) else g for g in geoms]
        else:
            wkb = [geometry.wkb if isinstance(geometry, BaseGeometry) else geometry]

        containment_mode = (
            ContainmentMode.IntersectsBoundary if buffer else ContainmentMode.ContainsCentroid
        )

        arr = wkb_to_cells(
            wkb,
            resolution=h3_resolution,
            containment_mode=containment_mode,
            flatten=True,
        )

        # arro3 vs pyarrow
        try:
            lst = arr.to_pylist()   # PyArrow
        except AttributeError:
            lst = list(arr)         # arro3

        # dedup mantendo ordem
        h3_indexes = list(dict.fromkeys(lst))
        return [h3.int_to_str(int(h)) for h in h3_indexes]

    def _h3_to_geoseries(self, h3_index):
        # <- AQUI estava o generator. Agora vira lista.
        if isinstance(h3_index, (str, int)):
            hlist = [h3_index]
        else:
            hlist = list(h3_index)

        # **lista** (não generator!)
        h3_int_indexes = [
            h if isinstance(h, int) else h3.str_to_int(h)
            for h in hlist
        ]

        return geopandas.GeoSeries.from_wkb(
            cells_to_wkb_polygons(h3_int_indexes),
            crs=4326,
        )

    def interpolate(self, h3_resolution: int = 9, buffer: bool = True):
        self.gdf = self.gdf.explode(index_parts=True).reset_index(drop=True)
        h3_list = self._shapely_geometry_to_h3(self.gdf["geometry"], h3_resolution, buffer=buffer)
        h3_list = list(dict.fromkeys(h3_list))
        return geopandas.GeoDataFrame(
            data={"h3": h3_list},
            geometry=self._h3_to_geoseries(h3_list),
            crs=4326,
        )
