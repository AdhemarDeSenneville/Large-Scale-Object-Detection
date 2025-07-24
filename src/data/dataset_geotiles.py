import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.ops import unary_union


def get_bestpixel_geometry(gdf, boundary, overlaps=None):
    """
    Compute non-overlapping pixel geometries within 'boundary' for each tile in 'gdf'.
    Adds 'pixel_geometry' column to 'gdf' and sets it as the active geometry.
    """
    if overlaps is None:
        overlaps = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1, 0.001]

    gdf['bestpixel_geometry'] = None
    remaining = sorted(gdf.index, key=lambda i: gdf.at[i, 'cloud_coverage'])
    #remaining = list(gdf.index)
    mapped = unary_union([])
    
    for threshold in overlaps:
        added = True
        while added:
            added = False
            for idx in remaining[:]:
                tile = gdf.loc[idx, 'geometry']
                diff = tile.difference(mapped)
                if diff.is_empty:
                    remaining.remove(idx)
                    continue
                if (diff.area / tile.area) * 100 < threshold:
                    continue
                pix = diff.intersection(boundary)
                if pix.is_empty or (pix.area / tile.area) * 100 < threshold:
                    continue
                gdf.at[idx, 'bestpixel_geometry'] = pix
                mapped = unary_union([mapped, tile])
                remaining.remove(idx)
                added = True

    gdf.set_geometry('bestpixel_geometry', inplace=True)
    return gdf


def plot_tiles_pixel(gdf, boundary=None):

    gdf_ = gdf.copy()
    gdf_['geometry'] = gdf_['pixel_geometry']
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_.plot(
        ax=ax,
        facecolor="blue",
        edgecolor="black",
        alpha=0.3,
        linewidth=2
    )
    if boundary is not None:
        gpd.GeoSeries([boundary], crs=gdf_.crs)\
           .plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)
    for idx, row in gdf_.iterrows():
        x, y = row.geometry.centroid.coords[0]
        ax.text(x, y, f"id {idx} \n {row.cloud_coverage:.2f}%  \n {row.acquisition_year} ", ha='center', va='center', fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_tiles_bestpixel(gdf, boundary=None, cmap_name='RdYlGn_r', figsize=(10, 10)):
    """
    Plot pixel geometries colored by cloud coverage using GeoPandas built-in column mapping.
    """
    gdf_ = gdf.copy()    
    gdf_['geometry'] = gdf_['bestpixel_geometry']

    fig, ax = plt.subplots(figsize=figsize)
    gdf_.plot(
        column='cloud_coverage',
        cmap=cmap_name,
        vmin=gdf_['cloud_coverage'].min(),
        vmax=gdf_['cloud_coverage'].max(),
        legend=False,
        alpha=0.3,
        linewidth=2,
        edgecolor='black',
        ax=ax
    )
    if boundary is not None:
        gpd.GeoSeries([boundary], crs=gdf_.crs).plot(
            ax=ax, facecolor='none', edgecolor='black', linewidth=2
        )
    for idx, row in gdf.iterrows():
        x, y = row.geometry.centroid.coords[0]
        ax.text(x, y, f"{row.cloud_coverage:.2f}%, {idx}",
                ha='center', va='center', fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
