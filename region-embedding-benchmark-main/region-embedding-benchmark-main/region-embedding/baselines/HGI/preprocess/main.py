import pandas as pd
import geopandas as gpd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from shapely import wkt
import networkx as nx
import h3
from h3ronpy import cells_to_string, grid_disk
from h3ronpy.vector import ContainmentMode, cells_to_wkb_polygons, wkb_to_cells
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from collections.abc import Iterable

from sklearn.preprocessing import LabelEncoder
import torch

import numpy as np
import pickle as pkl
import os

COLUMN_INDEX = "GEOID"
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

        # Garante LISTA concreta de WKB (nada de generator/Series)
        if isinstance(geometry, gpd.GeoSeries):
            wkb = list(geometry.to_wkb().to_numpy())
        elif isinstance(geometry, gpd.GeoDataFrame):
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

        # Converte para lista Python e remove duplicados mantendo ordem
        try:
            lst = arr.to_pylist()   # PyArrow
        except AttributeError:
            lst = list(arr)         # arro3

        h3_indexes = list(dict.fromkeys(lst))
        return [h3.int_to_str(int(h)) for h in h3_indexes]

    def _h3_to_geoseries(self, h3_index):
        if isinstance(h3_index, (str, int)):
            hlist = [h3_index]
        else:
            hlist = list(h3_index)

        h3_int_indexes = [
            h if isinstance(h, int) else h3.str_to_int(h)
            for h in hlist
        ]

        return gpd.GeoSeries.from_wkb(
            cells_to_wkb_polygons(h3_int_indexes),
            crs=4326,
        )

    def interpolate(self, h3_resolution: int = 9, buffer: bool = True):
        self.gdf = self.gdf.explode(index_parts=True).reset_index(drop=True)
        h3_list = self._shapely_geometry_to_h3(self.gdf["geometry"], h3_resolution, buffer=buffer)
        h3_list = list(dict.fromkeys(h3_list))  

        return gpd.GeoDataFrame(
            data={"h3": h3_list},
            geometry=self._h3_to_geoseries(h3_list),
            crs=4326,
        )

class Util:
    def __init__(self) -> None:
        pass

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        
        All args must be of equal length.    
        
        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6378137 * c
        return km

    @staticmethod
    def diagonal_length_min_box(min_box):
        x1, y1, x2, y2 = min_box
        pt1 = (x1, y1)
        pt2 = (x2, y1)
        pt4 = (x1, y2)

        dist12 = scipy.spatial.distance.euclidean(pt1, pt2)
        dist23 = scipy.spatial.distance.euclidean(pt1, pt4)
    
        return np.sqrt(dist12**2 + dist23**2)

    @staticmethod
    def intra_inter_region_transition(poi1, poi2, column=COLUMN_INDEX):
        if poi1[column] == poi2[column]:
            return 1
        else:
            return 0.5

class Preprocess():
    def __init__(self, pois_filename, boroughs_filename, emb_filename, h3=False) -> None:
        self.pois_filename = pois_filename
        self.boroughs_filename = boroughs_filename
        self.embedding_filename = emb_filename
        self.h3 = h3

    def _align_by_intersection(self, reg_key: str):
        self.pois[reg_key]   = self.pois[reg_key].astype(str)
        self.boroughs[reg_key] = self.boroughs[reg_key].astype(str)

        poi_regs = set(self.pois[reg_key].unique())
        bor_regs = set(self.boroughs[reg_key].unique())
        regions  = sorted(poi_regs & bor_regs)

        # poda e reindexa
        self.pois = self.pois[self.pois[reg_key].isin(regions)].reset_index(drop=True)
        self.boroughs = (self.boroughs.set_index(reg_key).loc[regions].reset_index())

        # region_id coerente
        reg2idx = {r:i for i,r in enumerate(regions)}
        self.region_id = [reg2idx[r] for r in self.pois[reg_key].tolist()]

        # agora compute TUDO com boroughs já reindexado:
        self.region_area = (self.boroughs.to_crs(3857).area/1e6).values

        import libpysal
        w = libpysal.weights.fuzzy_contiguity(self.boroughs.geometry)
        adj = w.to_adjlist(remove_symmetric=False)
        self.region_adjacency = adj[['focal','neighbor']].T.values

        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity
        mat = pd.crosstab(self.pois[reg_key], self.pois['fclass']).reindex(regions, fill_value=0)
        self.region_coarse_region_similarity = cosine_similarity(mat.values)


    def _read_poi_data(self):
      self.pois = pd.read_csv(self.pois_filename)
      self.pois["geometry"] = self.pois["geometry"].apply(wkt.loads)
      self.pois = gpd.GeoDataFrame(self.pois, geometry="geometry", crs="EPSG:4326")
      # garante Point
      self.pois["geometry"] = self.pois["geometry"].apply(
          lambda x: x if x.geom_type == "Point" else x.centroid
      )
    
    def _read_boroughs_data(self):
      self.boroughs = pd.read_csv(self.boroughs_filename)
      self.boroughs["geometry"] = self.boroughs["geometry"].apply(wkt.loads)
      self.boroughs = gpd.GeoDataFrame(self.boroughs, geometry="geometry", crs="EPSG:4326")

      if self.h3:
          self.boroughs = H3Interpolation(self.boroughs).interpolate(8)
          self.boroughs = self.boroughs.drop_duplicates(subset=['h3'], keep='first').reset_index(drop=True)

          # POIs × H3
          self.pois = gpd.sjoin(
              self.pois[["feature_id","category","fclass","geometry"]],
              self.boroughs[["h3","geometry"]],
              how="inner", predicate="intersects"
          ).reset_index(drop=True)
          reg_key = "h3" if self.h3 else "GEOID"
          self._align_by_intersection(reg_key)

          # filtra só h3 observados
          used = sorted(self.pois["h3"].unique().tolist())
          self.boroughs = self.boroughs[self.boroughs["h3"].isin(used)].reset_index(drop=True)

      else:
          # >>>>>>>>> ALTERE AQUI: traga GEOID no sjoin <<<<<<<<<
          self.pois = gpd.sjoin(
              self.pois[["feature_id","category","fclass","geometry"]],
              self.boroughs[["GEOID","geometry"]],
              how="inner", predicate="intersects"
          ).reset_index(drop=True)
          reg_key = "h3" if self.h3 else "GEOID"
          self._align_by_intersection(reg_key)
          # mantenha apenas GEOIDs observados e fixe ordem
          used = sorted(self.pois["GEOID"].unique().tolist())
          self.boroughs = self.boroughs[self.boroughs["GEOID"].isin(used)]
          self.boroughs = self.boroughs.sort_values("GEOID").reset_index(drop=True)

      self.n_regions = len(self.boroughs)

    def _read_embedding(self):
      # Carrega no CPU e garante que é Tensor "desanexado"
      state = torch.load(self.embedding_filename, map_location="cpu")
      emb_t = state["in_embed.weight"]
      if isinstance(emb_t, torch.nn.Parameter):
          emb_t = emb_t.detach()
      else:
          # Pode vir como Tensor com requires_grad=True
          emb_t = emb_t.detach()
      emb_np = emb_t.cpu().numpy()             # (vocab, dim)

      # Sanitiza fclass -> int e dentro do range
      self.pois["fclass"] = self.pois["fclass"].astype(int)
      max_idx = emb_np.shape[0] - 1
      self.pois["fclass"] = self.pois["fclass"].clip(lower=0, upper=max_idx)

      # Constrói lista de vetores (rápido e sem autograd)
      emb_list = [row.tolist() for row in emb_np]
      idx = self.pois["fclass"].to_numpy()
      self.embedding_array = [emb_list[i] for i in idx]

      # (opcional) manter a coluna no GeoDataFrame
      self.pois["embedding"] = self.embedding_array


    def _create_graph(self):
        if os.path.exists('/content/edges.csv'):
            self.edges = pd.read_csv('/content/edges.csv')
            return

        column = 'GEOID'
        if self.h3:
            column = "h3"

        print(self.pois)
        points = np.array(self.pois.geometry.apply(lambda x: [x.x, x.y]).tolist())
        D = Util.diagonal_length_min_box(self.pois.geometry.unary_union.envelope.bounds)

        triangles = scipy.spatial.Delaunay(points, qhull_options="QJ QbB Pp").simplices

        G = nx.Graph()
        G.add_nodes_from(range(len(points)))

        from itertools import combinations

        for simplex in triangles:
            comb = combinations(simplex, 2)
            for x, y in comb:
                if not G.has_edge(x, y):
                    dist = Util.haversine_np(*points[x], *points[y])
                    w1 = np.log((1+D**(3/2))/(1+dist**(3/2)))
                    w2 = Util.intra_inter_region_transition(
                        self.pois.iloc[x], 
                        self.pois.iloc[y],
                        column=column
                    )
                    G.add_edge(x, y, weight=w1*w2)
        
        self.edges = nx.to_pandas_edgelist(G)
        mi = self.edges['weight'].min()
        ma = self.edges['weight'].max()
        self.edges['weight'] = self.edges['weight'].apply(lambda x: (x-mi)/(ma-mi))

        self.edges.to_csv('/content/edges.csv', index=False)
    
    def _get_region_adjacency(self):
        import libpysal

        polygons_boroughs = self.boroughs.geometry
        adj = libpysal.weights.fuzzy_contiguity(polygons_boroughs)
        self.adj_list = adj.to_adjlist(remove_symmetric=False)
    
    def _get_region_id(self):
        # já foi definido em _align_by_intersection(...)
        assert hasattr(self, "region_id"), "region_id precisa ser definido no alinhamento"
        return
    def _get_coarse_region_similarity(self):
        # já calculado no alinhamento
        assert hasattr(self, "region_coarse_region_similarity"), "coarse sim precisa do alinhamento"
        return

    def get_data_torch(self):
      print("reading poi data")
      self._read_poi_data()

      print("reading boroughs data")
      self._read_boroughs_data()   # aqui dentro já houve o alinhamento e o cálculo de region_id/area/adj/sim
      print("reading boroughs data")

      self.pois = self.pois.reset_index(drop=True)
      self.pois["row_idx"] = range(len(self.pois))
      self.pois[["row_idx", "feature_id"]].to_csv("poi_index.csv", index=False)

      reg_key = "h3" if self.h3 else "GEOID"
      cols = ["row_idx", "feature_id"]
      if reg_key in self.pois.columns:
          cols.append(reg_key)
      self.pois[cols].to_csv("poi_region_map.csv", index=False)
      # <<<

      print("creating graph")
      self._create_graph()

      # estes viram NOOPs (já checados via assert acima)
      print("get region ids")
      self._get_region_id()

      print("reading embedding")
      self._read_embedding()

      print("creating region adjacency")
      self._get_region_adjacency()

      print("creating region similarity by cosine similarity of embeddings")
      self._get_coarse_region_similarity()

      print("finishing preprocessing")

      data = {}
      data['node_features'] = self.embedding_array
      data['edge_index'] = self.edges[["source", "target"]].T.values
      data['edge_weight'] = self.edges["weight"].values
      data['region_id'] = self.region_id
      data['coarse_region_similarity'] = self.region_coarse_region_similarity
      data['region_area'] = self.region_area
      data['region_adjacency'] = self.region_adjacency
      return data

 