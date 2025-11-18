import os
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame

import urllib3
urllib3.disable_warnings()
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*OpenSSL.*')


def get_parent_of_src(project_root: Path) -> Optional[Path]:
    """
    If `src` appears in `project_root` path (or any ancestor), return the parent directory of that `src`.
    Otherwise return None.
    """
    for p in (project_root, *project_root.parents):
        if p.name == "src":
            return p.parent
    return None

# Base directories
PROJECT_ROOT =  Path("/content/drive/MyDrive/MTL_POI_Novo")

DATA_ROOT = Path(os.environ.get('DATA_ROOT', PROJECT_ROOT / 'data'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', PROJECT_ROOT / 'data/output'))
RESULTS_ROOT = Path(os.environ.get('RESULTS_ROOT', PROJECT_ROOT / 'results'))

# Input data paths
IO_CHECKINS = DATA_ROOT / 'checkins'
if not IO_CHECKINS.exists():
    raise FileNotFoundError(f"Checkins directory not found: {IO_CHECKINS}")

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


class EmbeddingEngine(Enum):
    DGI = "dgi"
    HGI = "hgi"
    HMRM = "hmrm"


class Resources:
    """
    Static paths for resource files.
    """
    _miscellaneous_dir = DATA_ROOT / "miscellaneous"
    TL_AL: Path = _miscellaneous_dir / "tl_2022_01_tract_AL" / "tl_2022_01_tract.shp"
    TL_AZ: Path = _miscellaneous_dir / "tl_2022_04_tract_AZ" / "tl_2022_04_tract.shp"
    TL_GA: Path = _miscellaneous_dir / "tl_2022_13_tract_GA" / "tl_2022_13_tract.shp"
    TL_FL: Path = _miscellaneous_dir / "tl_2022_12_tract_FL" / "tl_2022_12_tract.shp"
    TL_CA: Path = _miscellaneous_dir / "tl_2022_06_tract_CA" / "tl_2022_06_tract.shp"
    TL_TX: Path = _miscellaneous_dir / "tl_2022_48_tract_TX" / "tl_2022_48_tract.shp"



class _DGIIoPath:
    # File name constants
    CTA_FILE: str = "cta.csv"
    INTER_DATA_FILE: str = "data_inter.pkl"

    # DGI embeddings output
    _dgi_dir: Path = OUTPUT_DIR / EmbeddingEngine.DGI.value

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the DGI output directory for a specific state."""
        return cls._dgi_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_temp_dir(cls, state: str) -> Path:
        """Get the temp directory for a specific state."""
        return cls.get_output_dir(state) / "temp"

    @classmethod
    def get_inter_file(cls, state: str) -> Path:
        """Get the intermediate data file path for a specific state."""
        return cls.get_temp_dir(state) / cls.INTER_DATA_FILE

    @classmethod
    def get_cta_file(cls, state: str) -> Path:
        """Get the CTA file path for a specific state."""
        return cls.get_temp_dir(state) / cls.CTA_FILE


class _HGIIoPath:
    # File name constants
    POIS_FILE: str = "pois_gowalla.csv"
    BOROUGHS_FILE: str = "boroughs_area.csv"
    POI_EMB_FILE: str = "poi-encoder.tensor"
    GRAPH_DATA_FILE: str = "gowalla.pt"
    SECOND_CLASS_WALKS_FILE: str = "second_class_walks.pkl"
    POI_INDEX_FILE: str = "poi_index.csv"
    POI_REGION_MAP_FILE: str = "poi_region_map.csv"
    EDGES_FILE: str = "edges.csv"
    POIS_PROCESSED_FILE: str = "pois.csv"

    # HGI embeddings output
    _hgi_dir: Path = OUTPUT_DIR 

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the HGI output directory for a specific state."""
        return cls._hgi_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_temp_dir(cls, state: str) -> Path:
        """Get the temp directory for a specific state."""
        return cls.get_output_dir(state) / "temp"

    @classmethod
    def get_pois_file(cls, state: str) -> Path:
        """Get the POIs file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POIS_FILE

    @classmethod
    def get_boroughs_file(cls, state: str) -> Path:
        """Get the boroughs file path for a specific state."""
        return cls.get_temp_dir(state) / cls.BOROUGHS_FILE

    @classmethod
    def get_poi_emb_file(cls, state: str) -> Path:
        """Get the POI embeddings file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POI_EMB_FILE

    @classmethod
    def get_graph_data_file(cls, state: str) -> Path:
        """Get the graph data file path for a specific state."""
        return cls.get_temp_dir(state) / cls.GRAPH_DATA_FILE

    @classmethod
    def get_second_class_walks_file(cls, state: str) -> Path:
        """Get the second class walks file path for a specific state."""
        return cls.get_temp_dir(state) / cls.SECOND_CLASS_WALKS_FILE

    @classmethod
    def get_poi_index_file(cls, state: str) -> Path:
        """Get the POI index file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POI_INDEX_FILE

    @classmethod
    def get_poi_region_map_file(cls, state: str) -> Path:
        """Get the POI region map file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POI_REGION_MAP_FILE

    @classmethod
    def get_edges_file(cls, state: str) -> Path:
        """Get the edges file path for a specific state."""
        return cls.get_temp_dir(state) / cls.EDGES_FILE

    @classmethod
    def get_pois_processed_file(cls, state: str) -> Path:
        """Get the processed POIs file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POIS_PROCESSED_FILE
class IoPaths:
    """
    Static paths for I/O operations.
    """
    # agora usamos CSV
    EMBEDDINGS_FILE: str = "embeddings.csv"

    DGI = _DGIIoPath
    HGI = _HGIIoPath

    # ========== EMBEDDINGS GERAIS ==========

    @classmethod
    def get_embedd(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        # /content/drive/.../data/output/montana/embeddings.csv
        return OUTPUT_DIR / state.lower() / cls.EMBEDDINGS_FILE

    @classmethod
    def load_embedd(cls, state: str, embedd_engine: EmbeddingEngine) -> DataFrame:
        return pd.read_csv(cls.get_embedd(state, embedd_engine))

    # ========== CHECKINS DA CIDADE ==========

    @classmethod
    def get_city(cls, state: str, ext: str = 'csv') -> Path:
        # /content/drive/.../data/checkins/Montana.csv
        return DATA_ROOT / 'checkins' / f"{state.capitalize()}.{ext}"

    @classmethod
    def load_city(cls, state: str, ext: str = 'csv') -> DataFrame:
        return pd.read_csv(cls.get_city(state, ext))

    # ========== INPUTS PARA MTL ==========

    @classmethod
    def get_input_dir(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        # /content/drive/.../data/output/montana/pre-processing
        return OUTPUT_DIR / state.lower() / "pre-processing"

    @classmethod
    def get_category(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        # /.../data/output/montana/pre-processing/category-input.csv
        return cls.get_input_dir(state, embedd_engine) / "category-input.csv"

    @classmethod
    def load_category(cls, state: str, embedd_engine: EmbeddingEngine) -> DataFrame:
        return pd.read_csv(cls.get_category(state, embedd_engine))

    @classmethod
    def get_next(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        # /.../data/output/montana/pre-processing/next-input.csv
        return cls.get_input_dir(state, embedd_engine) / "next-input.csv"

    @classmethod
    def load_next(cls, state: str, embedd_engine: EmbeddingEngine) -> DataFrame:
        return pd.read_csv(cls.get_next(state, embedd_engine))

    # ========== SEQUÊNCIAS (NEXT) ==========

    @classmethod
    def get_seq_next(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        # /.../data/output/montana/temp/poi-sequences.csv
        return OUTPUT_DIR / state.lower() / "temp" / "poi-sequences.csv"

    # ========== RESULTADOS / FOLDS ==========

    @classmethod
    def get_results_dir(cls, state: str, enbedd_engine: EmbeddingEngine) -> Path:
        # /.../results/montana
        return RESULTS_ROOT / state.lower()

    @classmethod
    def get_folds_dir(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        # /.../data/output/montana/folds
        return OUTPUT_DIR / state.lower() / "folds"

