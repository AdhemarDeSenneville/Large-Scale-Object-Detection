import geopandas as gpd
from shapely.geometry import box

class EuropeCountryContours:
    def __init__(self, folder_path: str = '/home/adhemar/Bureau/datasets/Coutry_Shapes/ne_10m_admin_0_countries_lakes/ne_10m_admin_0_countries_lakes.shp'):
        """
        Initialize with the path to the folder containing the Natural Earth shapefile.
        """
        self.gdf = gpd.read_file(f"{folder_path}") #/ne_10m_admin_0_countries.shp

        self.gdf = self.gdf.to_crs(epsg=4326)
        
        # Define a rectangular “Europe box” in lon/lat
        # [minx, miny, maxx, maxy] = [west, south, east, north]
        self.europe_box = box(-25.0, 34.0, 45.0, 72.0)

    def get_country_shape(
        self,
        country_name: str,
        epsg: int = 4326,
        in_europe: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Returns just the named country’s outline. If in_europe=True,
        any geometry outside the Europe box is clipped away.
        """
        # 1) pick out the country by name (case‐insensitive partial match)
        df = self.gdf[self.gdf['NAME'].str.contains(country_name, case=False, na=False)]
        if df.empty:
            raise ValueError(f"Country '{country_name}' not found in shapefile.")

        # 2) optionally clip to Europe
        if in_europe:
            df = df.copy()  # avoid SettingWithCopyWarning
            # perform geometry intersection
            df['geometry'] = df.geometry.intersection(self.europe_box)
            # drop any empty geometries
            df = df[~df.is_empty]
            if df.empty:
                raise ValueError(
                    f"After clipping to Europe, no geometry remains for '{country_name}'."
                )

        # 3) reproject to target CRS and return
        return df.to_crs(epsg=epsg)