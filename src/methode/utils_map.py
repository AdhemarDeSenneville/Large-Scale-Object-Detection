import folium
import numpy as np
from pyproj import Transformer

from scipy.stats import gaussian_kde
import geopandas as gpd


def remove_multy_polygone(shape_object):
    
    def get_largest_polygon(geometry):
        if geometry.geom_type == "MultiPolygon":
            return max(geometry.geoms, key=lambda g: g.area) 
        return geometry
    
    shape_object["geometry"] = shape_object["geometry"].apply(get_largest_polygon)
    return shape_object


def get_heat_map(
        position_known, 
        heatmap_bw_method = 'scott',
        heatmap_resolution = 500,
        multiplier = 1,
    ):
    position_known_ = np.array(position_known).T
    kde = gaussian_kde(position_known_, bw_method=heatmap_bw_method) # 'scott' bw_method=heatmap_bw_method, # BE CARFULL WITH THE METHODE !!! 

    # Create grid for density estimation
    uncrop_dist = 100000
    x_min_heatmap, x_max_heatmap = position_known_[0,:].min() - uncrop_dist, position_known_[0,:].max() + uncrop_dist
    y_min_heatmap, y_max_heatmap = position_known_[1,:].min() - uncrop_dist, position_known_[1,:].max() + uncrop_dist

    heatmap_resolution_x = int((x_max_heatmap-x_min_heatmap)/heatmap_resolution)
    heatmap_resolution_y = int((y_max_heatmap-y_min_heatmap)/heatmap_resolution)

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min_heatmap, x_max_heatmap, heatmap_resolution_x), 
        np.linspace(y_max_heatmap, y_min_heatmap, heatmap_resolution_y)
    )
    density_values = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
    print('MAX Value',np.max(density_values))
    density_values = np.clip(density_values / np.max(density_values) * multiplier, 0, 1)

    print("KDE Covariance :", kde.covariance)
    print("X limits:", x_min_heatmap, x_max_heatmap)
    return density_values, [x_min_heatmap, x_max_heatmap, y_max_heatmap, y_min_heatmap]


def build_map(
    all_new_detections,
    all_old_detections,
    filtered_position,
    save_path = "map_detections.html",
    
    center_lat = 46,
    center_lon = 2.5,
    is_google_map = False,
    is_obb = True,
    is_annotation = False,
    is_heat_map = False,
    category_colors = {1: "red", 2: "green", 3: "blue"},
):
    # Define category colors
    
    # MAP Object
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

    # MAP Back Ground
    if is_google_map:
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Satellite",
            overlay=True
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
        ).add_to(m)


    # Add Markers
    def add_markers(data, indexes, color, label):
        for i in indexes:

            score = data[i]['global_score']
            lon, lat = data[i]['gps']

            folium.Marker(
                location=[lat, lon],
                popup=f"{label}: ({lat:.5f}, {lon:.5f}), Score: {score}",
                icon=folium.Icon(color=color)
            ).add_to(m)

    # Add markers to the map
    add_markers(all_new_detections, filtered_position['detected_known_new'], "green", "Known Detected")
    add_markers(all_new_detections, filtered_position['detected_unknown_new'], 'orange', "Unknown Detected")
    add_markers(all_old_detections, filtered_position['undetected_known_old'], "black", "Known Undetected")

    # Function to add bounding boxes to map
    def add_bboxes(data, indexes, line_style):

        for i in indexes:

            list_category = [data[i]["category_id"]] + [detec["category_id"] for detec in data[i]['inside_list']]
            list_score = [data[i]["score"]] + [detec["score"] for detec in data[i]['inside_list']]
            
            if is_obb:
                list_obb_points = [data[i]["segmentation"][0]] + [detec["segmentation"][0] for detec in data[i]['inside_list']]
                
                
                for detec_index in range(len(list_obb_points)):

                    obb_points = list_obb_points[detec_index]
                    category = list_category[detec_index]
                    score = list_score[detec_index]
                
                    obb_pairs = [(obb_points[i], obb_points[i + 1]) for i in range(0, len(obb_points), 2)]
                    latlon_points = [transformer.transform(x, y) for x, y in obb_pairs]
                    latlon_points = [[lat, lon] for lon, lat in latlon_points]

                    # Create a Polygon for the OBB
                    folium.Polygon(
                        locations=latlon_points,
                        color=category_colors.get(category, "black"),
                        fill=True,
                        fill_opacity=0.1,
                        dash_array=line_style,
                        popup=f"Score: {score}"
                    ).add_to(m)
                    
            else:
                x, y, w, h = data[i]["bbox"]
                category = data[i]["category_id"]
                score = data[i]["score"]
                
                # Convert all four corners to lat/lon
                x_min, y_min = transformer.transform(x, y)
                x_max, y_max = transformer.transform(x + w, y + h)

                # Create a rectangle with different styles
                folium.Rectangle(
                    bounds=[[y_min, x_min], [y_max, x_max]],
                    color=category_colors.get(category, "black"),
                    fill=True,
                    fill_opacity=0.1,
                    dash_array=line_style,
                    popup=f"Score: {score}"
                ).add_to(m)

    if is_annotation:
        add_bboxes(all_new_detections, filtered_position['detected_unknown_new'], None)
        #add_bboxes(flat_results_gt, "5,5")


    if is_heat_map:
        try: # Overlay heatmap on Folium
            image_overlay = ImageOverlay(
                name="Heatmap",
                image="heatmap.png",
                bounds=[[y_min_heatmap, x_min_heatmap], [y_max_heatmap, x_max_heatmap]],
                opacity=0.5,
                interactive=True
            )
            image_overlay.add_to(m)
        except:
            print('No Heatmap')


    folium.LayerControl().add_to(m)

    if True:

        from branca.element import Template, MacroElement

        # Define the legend HTML and CSS
        legend_html = """
        <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            font-size: 14px;
        ">
            <b>Legend</b>
            <br>
            <i class="fa fa-map-marker fa-2x" style="color: green"></i> Known Detected
            <br>
            <i class="fa fa-map-marker fa-2x" style="color: orange"></i> Unknown Detected
            <br>
            <i class="fa fa-map-marker fa-2x" style="color: black"></i> Known Undetected
            <br>
            <div style="width: 20px; height: 10px; background: red; display: inline-block;"></div> Category 1
            <br>
            <div style="width: 20px; height: 10px; background: green; display: inline-block;"></div> Category 2
            <br>
            <div style="width: 20px; height: 10px; background: blue; display: inline-block;"></div> Category 3
        </div>
        """

        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)

    m.save(save_path)