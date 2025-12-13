"""
Geo Lookup Module

Provides lat/lon to region mapping with broadcast variable optimization.
"""

import json
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


class GeoLookup:
    """
    Geo enrichment using lat/lon to region mapping.
    
    Uses broadcast variables for efficient distributed lookup.
    """
    
    def __init__(
        self,
        spark: SparkSession,
        config_path: Optional[str] = None,
        default_region: str = "UNKNOWN"
    ):
        """
        Initialize GeoLookup.
        
        Args:
            spark: SparkSession instance
            config_path: Path to geo_regions.json file
            default_region: Default region for unmapped coordinates
        """
        self.spark = spark
        self.default_region = default_region
        self.regions = []
        self.city_overrides = {}
        
        if config_path:
            self._load_config(config_path)
        else:
            self._load_default_regions()
        
        # Broadcast the lookup data
        self._broadcast_regions = spark.sparkContext.broadcast(self.regions)
        self._broadcast_cities = spark.sparkContext.broadcast(self.city_overrides)
    
    def _load_config(self, config_path: str) -> None:
        """Load region config from JSON file."""
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                self.regions = data.get("regions", [])
                self.city_overrides = data.get("city_overrides", {})
        else:
            print(f"Warning: Geo config not found at {config_path}, using defaults")
            self._load_default_regions()
    
    def _load_default_regions(self) -> None:
        """Load hardcoded default regions."""
        self.regions = [
            {"name": "North America", "code": "NA", 
             "bounds": {"min_lat": 15.0, "max_lat": 72.0, "min_lon": -170.0, "max_lon": -50.0}},
            {"name": "Europe", "code": "EU", 
             "bounds": {"min_lat": 35.0, "max_lat": 72.0, "min_lon": -25.0, "max_lon": 65.0}},
            {"name": "Asia", "code": "AS", 
             "bounds": {"min_lat": -10.0, "max_lat": 77.0, "min_lon": 65.0, "max_lon": 180.0}},
            {"name": "South America", "code": "SA", 
             "bounds": {"min_lat": -56.0, "max_lat": 13.0, "min_lon": -82.0, "max_lon": -34.0}},
            {"name": "Africa", "code": "AF", 
             "bounds": {"min_lat": -35.0, "max_lat": 37.0, "min_lon": -18.0, "max_lon": 52.0}},
            {"name": "Oceania", "code": "OC", 
             "bounds": {"min_lat": -50.0, "max_lat": 0.0, "min_lon": 110.0, "max_lon": 180.0}},
        ]
        self.city_overrides = {
            "New York": {"region": "NA"},
            "Los Angeles": {"region": "NA"},
            "London": {"region": "EU"},
            "Paris": {"region": "EU"},
            "Tokyo": {"region": "AS"},
            "Sydney": {"region": "OC"},
        }


def lookup_region_by_coords(lat: float, lon: float, regions: list, default: str) -> str:
    """
    Look up region code by lat/lon coordinates.
    
    This is a pure Python function for use in UDFs.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        regions: List of region dicts with bounds
        default: Default region code
    
    Returns:
        Region code (e.g., "NA", "EU", "AS")
    """
    if lat is None or lon is None:
        return default
    
    for region in regions:
        bounds = region.get("bounds", {})
        if (bounds.get("min_lat", -90) <= lat <= bounds.get("max_lat", 90) and
            bounds.get("min_lon", -180) <= lon <= bounds.get("max_lon", 180)):
            return region.get("code", default)
    
    return default


def lookup_region_by_city(city: str, city_overrides: dict, default: str) -> str:
    """
    Look up region by city name (fast path).
    
    Args:
        city: City name
        city_overrides: Dict of city -> region mappings
        default: Default region
    
    Returns:
        Region code
    """
    if city is None:
        return default
    
    # Normalize city name
    city_normalized = city.strip().title()
    
    if city_normalized in city_overrides:
        return city_overrides[city_normalized].get("region", default)
    
    return None  # None means fallback to coords lookup


def create_geo_enrichment_udf(
    regions: list,
    city_overrides: dict,
    default_region: str = "UNKNOWN"
):
    """
    Create a UDF for geo enrichment.
    
    Args:
        regions: Broadcast value of region list
        city_overrides: Broadcast value of city overrides
        default_region: Default region code
    
    Returns:
        PySpark UDF
    """
    def enrich(city: str, lat: float, lon: float) -> str:
        # Try city lookup first (fast path)
        if city:
            city_region = lookup_region_by_city(city, city_overrides, default_region)
            if city_region:
                return city_region
        
        # Fall back to coordinate lookup
        return lookup_region_by_coords(lat, lon, regions, default_region)
    
    return F.udf(enrich, StringType())


def enrich_dataframe_with_geo(
    df: DataFrame,
    spark: SparkSession,
    city_col: str = "city",
    lat_col: str = "lat",
    lon_col: str = "lon",
    output_col: str = "region",
    config_path: Optional[str] = None,
    default_region: str = "UNKNOWN"
) -> DataFrame:
    """
    Enrich DataFrame with geo region based on city/coordinates.
    
    This is the main entry point for geo enrichment.
    
    Args:
        df: Input DataFrame
        spark: SparkSession
        city_col: Column name for city
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        output_col: Output column name for region
        config_path: Path to geo config JSON
        default_region: Default region for unmapped locations
    
    Returns:
        DataFrame with region column added
    """
    geo = GeoLookup(spark, config_path, default_region)
    
    # Get broadcast values
    regions = geo._broadcast_regions.value
    cities = geo._broadcast_cities.value
    
    # Create UDF
    geo_udf = create_geo_enrichment_udf(regions, cities, default_region)
    
    # Handle missing columns gracefully
    if lat_col not in df.columns:
        df = df.withColumn(lat_col, F.lit(None).cast("double"))
    if lon_col not in df.columns:
        df = df.withColumn(lon_col, F.lit(None).cast("double"))
    if city_col not in df.columns:
        df = df.withColumn(city_col, F.lit(None).cast("string"))
    
    # Apply enrichment
    df = df.withColumn(
        output_col,
        geo_udf(F.col(city_col), F.col(lat_col), F.col(lon_col))
    )
    
    return df
