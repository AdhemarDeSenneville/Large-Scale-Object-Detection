import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import shapely.geometry
import cv2
import numpy as np
import geopandas as gpd
from collections import Counter
import time
from pyproj import Transformer
from tkinter import filedialog, messagebox


def count_annotations(positions, new_column='Human_Check_Spot_2023'):
    # Counter to track how many of each annotation value we see
    annotation_counts = Counter()
    
    for feat in positions:
        annotation = feat['properties']['human_feedback'].get(new_column)
        if annotation is not None:
            annotation_counts[annotation] += 1
    
    total = sum(annotation_counts.values())
    print(f'Number of annotations: {total} out of {len(positions)}')
    print(f"Remaining : {len(positions) - total}")
    
    if annotation_counts:
        print('Breakdown by annotation value:')
        for annotation_value, count in annotation_counts.items():
            print(f'  {annotation_value}: {count}')
    else:
        print('No annotations found.')

def annotate_positions(
        dataset, 
        positions, 
        path_export_example_images,
        resume_annotation = False, 
        geo_indexing = True,
        annotation_name = 'Human_Check_Spot_2023',
        config_gui = {},
        format = 'geojson',
        resolution_for_clicks = 1.5,
    ):
    print('Not implemented export image')
    
    window_size = config_gui.get("window_size", 1000)
    zoom_window_size = config_gui.get("window_zoom_size", 400)
    display_size = config_gui.get("display_sizes", window_size)  # Available display sizes

    if geo_indexing:
        if format == 'geopandas':
            positions = dataset.geo_index_gps_positions(positions, format = 'geopandas')
        elif format == 'geojson':
            positions = dataset.geo_index_gps_positions(positions, format = 'geojson')
    
    if format == 'geopandas':
        raise ValueError("The format 'geopandas' is not supported in this function. Use 'geojson' instead.")
        if new_column not in positions.columns:
            positions[new_column] = None 

        if resume_annotation:
            unannotated_indices = positions[positions[new_column].isnull()].index.to_list()
        else:
            unannotated_indices = positions.index.to_list()
    

    elif format == 'geojson':
        for index, detec in enumerate(positions):
            if 'human_feedback' not in detec['properties'].keys():
                detec['properties']['human_feedback'] = {}

        if resume_annotation:
            unannotated_indices = []
            for index, detec in enumerate(positions):   
                if annotation_name not in detec['properties']['human_feedback'].keys():
                    unannotated_indices.append(index)
        else:
            unannotated_indices = list(range(len(positions)))
    

    index = 0
    root = tk.Tk()
    root.title("Image Annotation")

    # Create main frames for layout
    frame_left = tk.Frame(root)
    frame_left.grid(row=0, column=0, padx=10, pady=10)

    frame_center = tk.Frame(root)
    frame_center.grid(row=0, column=1, padx=10, pady=10)

    frame_right = tk.Frame(root)
    frame_right.grid(row=0, column=2, padx=10, pady=10)

    img_label = tk.Label(frame_center)
    img_label.pack()

    zoom_img_label = tk.Label(frame_right)
    zoom_img_label.pack()

    root.current_raw_img = None
    root.current_ann_img = None

    def show_image(idx, center_xy=None):
        """Read the image window for positions[idx] and show it in the Tk labels."""
        loading_label = tk.Label(root, text="Loading image...", fg="red", font=("Arial", 24, "bold"))
        loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Centered in the window
        root.update_idletasks()

        if format == 'geopandas':
            x, y = positions.iloc[idx].geometry.x, positions.iloc[idx].geometry.y
        elif format == 'geojson':
            x, y = positions[idx]['geometry']['coordinates'][0], positions[idx]['geometry']['coordinates'][1]
        
        print('POSITIONS', x, y, '| GPS', *reversed(Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform(x, y)))

        max_retries = 5
        for attempt in range(1, max_retries+1):
            try:
                img_array = dataset.get_rgb_image(x, y, window_size=window_size)
                break   # success!
            except TypeError as e:
                print(f"[Attempt {attempt}/{max_retries}] download failed: {e}. retrying in 2s…")
                time.sleep(3)
        else:
            print(f"❌ Could not fetch image at ({x},{y}) after {max_retries} retries.")
            return

        # Zoom in the centre
        h, w, _ = img_array.shape
        crop_size = zoom_window_size // 2
        center_x, center_y = w // 2, h // 2
        zoom_img_array = img_array[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]

        pil_img = Image.fromarray(img_array).resize((display_size, display_size), Image.LANCZOS)
        zoom_pil_img = Image.fromarray(zoom_img_array).resize((display_size, display_size), Image.LANCZOS)

        root.current_raw_img = pil_img
        root.current_ann_img = zoom_pil_img

        loading_label.destroy()

        # Draw a red point in the center of both images
        draw = ImageDraw.Draw(pil_img)
        draw_zoom = ImageDraw.Draw(zoom_pil_img)

        if center_xy is None:
            center_x = display_size // 2
            center_y = display_size // 2
            color = "red"
        else:
            center_x = center_xy[0]
            center_y = center_xy[1]
            color = "green"

        center_x_zoom = int(display_size // 2 + (center_x - display_size // 2) * window_size / zoom_window_size)
        center_y_zoom = int(display_size // 2 + (center_y - display_size // 2) * window_size / zoom_window_size)

        ellipse_size = 2
        draw.ellipse((center_x - ellipse_size, center_y - ellipse_size, center_x + ellipse_size, center_y + ellipse_size), fill=color)
        draw_zoom.ellipse((center_x_zoom - ellipse_size, center_y_zoom - ellipse_size, center_x_zoom + ellipse_size, center_y_zoom + ellipse_size), fill=color)

        # Display images
        img_label.img_ref = ImageTk.PhotoImage(pil_img)
        img_label.configure(image=img_label.img_ref)

        zoom_img_label.img_ref = ImageTk.PhotoImage(zoom_pil_img)
        zoom_img_label.configure(image=zoom_img_label.img_ref)

        if format == 'geopandas':
            root.title(f"Index {idx+1}/{len(positions)}: x={x}, y={y}")
        elif format == 'geojson':
            source = ' '.join(positions[idx]['properties'].keys())
            root.title(f"Index {idx+1}/{len(positions)}: x={x}, y={y}, {source}")


    def go_next():
        """Advance to the next unannotated position or close if done."""
        nonlocal index
        index += 1

        print('ADVENCEMENT',index,"/",len(unannotated_indices))
        if index < len(unannotated_indices):
            show_image(unannotated_indices[index])
        else:
            root.destroy()
    
    def go_prev():
        """Advance to the next unannotated position or close if done."""
        nonlocal index
        index -= 1

        print('ADVENCEMENT',index,"/",len(unannotated_indices))
        if index > 0:
            show_image(unannotated_indices[index])
        else:
            root.destroy()

    
    def on_image_click(event):
        # 1) Convert the click location from display coords to full-size coords
        scale_main = window_size / float(display_size)
        new_center_x = int(event.x * scale_main)
        new_center_y = int(event.y * scale_main)

        if format == 'geopandas':
            old_x = positions.iloc[unannotated_indices[index]].geometry.x
            old_y = positions.iloc[unannotated_indices[index]].geometry.y
        elif format == 'geojson':
            old_x = positions[unannotated_indices[index]]['geometry']['coordinates'][0]
            old_y = positions[unannotated_indices[index]]['geometry']['coordinates'][1]


        dx_display = event.x - (display_size // 2)
        dy_display = event.y - (display_size // 2)

        # Scale offset back to full-size pixels.
        scale_main = window_size / float(display_size)
        dx_full = dx_display * scale_main
        dy_full = dy_display * scale_main

        # Convert pixel offset to meters (1.5 m per pixel).
        dx_m = dx_full * resolution_for_clicks
        dy_m = dy_full * resolution_for_clicks

        new_x = old_x + dx_m
        new_y = old_y - dy_m

        # Update the geometry in the DataFrame
        if format == 'geopandas':
            positions.at[unannotated_indices[index], 'geometry'] = shapely.geometry.Point(new_x, new_y)
        elif format == 'geojson':
            positions[unannotated_indices[index]]['geometry']['coordinates'][0] = new_x
            positions[unannotated_indices[index]]['geometry']['coordinates'][1] = new_y

        show_image(unannotated_indices[index])

        print(f"New center in pixel coordinates: ({new_center_x}, {new_center_y})")
        print(f"Old Coordinates: ({old_x}, {old_y})")
        print(f"New Coordinates: ({new_x}, {new_y})")

    # Define callbacks
    def on_present():
        positions[unannotated_indices[index]]['properties']['human_feedback'][annotation_name] = 'True'
        go_next()

    def on_not_present():
        positions[unannotated_indices[index]]['properties']['human_feedback'][annotation_name] = 'False'
        go_next()

    def on_unclear():
        positions[unannotated_indices[index]]['properties']['human_feedback'][annotation_name] = 'Unclear'
        go_next()

    def on_construction():
        positions[unannotated_indices[index]]['properties']['human_feedback'][annotation_name] = 'Construction'
        go_next()
    
    def quit_app():
        root.destroy()

    def save_img():
        raw = root.current_raw_img
        ann = root.current_ann_img
        if raw is None or ann is None:
            messagebox.showwarning("Nothing to save", "No image is currently loaded.")
            return

        # combine them horizontally
        w, h = raw.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(raw, (0, 0))
        combined.paste(ann, (w, 0))

        # ask the user where to save
        default_name = f"{unannotated_indices[index]:03d}.png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialdir=path_export_example_images,
            initialfile=default_name
        )
        if not path:
            return

        try:
            combined.save(path, format="PNG")
            messagebox.showinfo("Saved", f"Image successfully saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error saving", str(e))

    img_label.bind("<Button-1>", on_image_click)
    #  Create buttons in the right column under the zoomed image
    btn_present = tk.Button(frame_left, text="Object", bg="green", fg="white", command=on_present)
    btn_not_present = tk.Button(frame_left, text="Background", bg="red", fg="white", command=on_not_present)
    btn_unclear = tk.Button(frame_left, text="Unclear", bg="orange", fg="white", command=on_unclear)
    btn_construction = tk.Button(frame_left, text="Construction", bg="yellow", fg="black", command=on_construction)
    btn_next = tk.Button(frame_left, text="Next ▶", bg="blue", fg="white", command=go_next)  # NEW BUTTON
    btn_prev = tk.Button(frame_left, text="Prev ◀", bg="blue", fg="white", command=go_prev)  # NEW BUTTON
    ptn_export = tk.Button(frame_left, text="Export", bg="blue", fg="white", command=save_img)  # NEW BUTTON
    btn_quit = tk.Button(frame_left, text="Quit", bg="gray", fg="white", command=quit_app)

    btn_present.pack(fill=tk.X, pady=2)
    btn_not_present.pack(fill=tk.X, pady=2)
    tk.Label(frame_left, text="").pack(expand=True)
    btn_unclear.pack(fill=tk.X, pady=2)
    btn_construction.pack(fill=tk.X, pady=2)
    tk.Label(frame_left, text="").pack(expand=True)
    tk.Label(frame_left, text="").pack(expand=True)
    ptn_export.pack(fill=tk.X, pady=5)
    btn_next.pack(fill=tk.X, pady=5)
    btn_prev.pack(fill=tk.X, pady=5)
    btn_quit.pack(fill=tk.X, pady=5)

    if len(positions) > 0:
        show_image(unannotated_indices[index])
        root.mainloop()

    """
    if format == 'geopandas':
        positions.loc[labels_dict["present"], 'human_feedback'] = 'True'
        positions.loc[labels_dict["not_present"], 'human_feedback'] = 'False'
        positions.loc[labels_dict["unclear"], 'human_feedback'] = 'Unclear'
        positions.loc[labels_dict["construction"], 'human_feedback'] = 'Construction'
    elif format == 'geojson':
        for index in labels_dict["present"]:
            positions[index]['properties']['human_feedback'][annotation_name] = 'True'
        for index in labels_dict["not_present"]:
            positions[index]['properties']['human_feedback'][annotation_name] = 'False'
        for index in labels_dict["unclear"]:
            positions[index]['properties']['human_feedback'][annotation_name] = 'Unclear'
        for index in labels_dict["construction"]:
            positions[index]['properties']['human_feedback'][annotation_name] = 'Construction'
    """
    return positions


def annotate_detections(
        dataset, 
        detections, 
        path_export_example_images,
        resume_annotation = False, 
        geo_indexing = True,
        annotation_name = 'Human_Check_Spot_2023', 
        config_gui = {}
    ):
    
    uncrop_size = config_gui.get("uncrop_size", 100)
    display_size = config_gui.get("display_sizes", 900) 
    annotation_threshold = config_gui.get("annotation_threshold", 0)

    if geo_indexing:
        detections = dataset.geo_index_gps_positions(detections, format = 'old')
    
    if resume_annotation:
        unannotated_indices = []
        for index, detec in enumerate(detections):

            if 'human_feedback' not in detec.keys():
                detec['human_feedback'] = {}
            
            if annotation_name not in detec['human_feedback'].keys():
                unannotated_indices.append(index)
    else:
        unannotated_indices = list(range(len(detections)))
    
    index = 0
    root = tk.Tk()
    root.title("Image Annotation")

    # Create main frames for layout
    frame_left = tk.Frame(root)
    frame_left.grid(row=0, column=0, padx=10, pady=10)

    frame_center = tk.Frame(root)
    frame_center.grid(row=0, column=1, padx=10, pady=10)

    frame_right = tk.Frame(root)
    frame_right.grid(row=0, column=2, padx=10, pady=10)

    img_label = tk.Label(frame_center)
    img_label.pack()

    ann_img_label = tk.Label(frame_right)
    ann_img_label.pack()


    root.current_raw_img = None
    root.current_ann_img = None


    def show_image(idx):
        """Read the image window for positions[idx] and show it in the Tk labels."""
        loading_label = tk.Label(root, text="Loading image...", fg="red", font=("Arial", 24, "bold"))
        loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Centered in the window
        root.update_idletasks()

        x, y = detections[idx]['position']
        window_size = max(detections[idx]['bbox'][2], detections[idx]['bbox'][3])
        window_size += uncrop_size

        img_array = dataset.get_rgb_image(x, y, window_size=window_size)
        img_array_annotated = annotate_image(img_array, detections[idx], x, y, window_size, 1.5, annotation_threshold)

        pil_img = Image.fromarray(img_array).resize((display_size, display_size), Image.LANCZOS)
        ann_pil_img = Image.fromarray(img_array_annotated).resize((display_size, display_size), Image.LANCZOS)

        root.current_raw_img = pil_img
        root.current_ann_img = ann_pil_img

        loading_label.destroy()
        # Display images
        img_label.img_ref = ImageTk.PhotoImage(pil_img)
        img_label.configure(image=img_label.img_ref)

        ann_img_label.img_ref = ImageTk.PhotoImage(ann_pil_img)
        ann_img_label.configure(image=ann_img_label.img_ref)

        root.title(f"Index {idx+1}/{len(detections)}: x={x}, y={y}")
    
    def quit_app():
        root.destroy()

    def go_next():
        """Advance to the next unannotated position or close if done."""
        nonlocal index
        index += 1

        print('ADVENCEMENT',index,"/",len(unannotated_indices))
        if index < len(unannotated_indices):
            show_image(unannotated_indices[index])
        else:
            root.destroy()

    def on_present():
        detections[unannotated_indices[index]]['human_feedback'][annotation_name] = 'True'
        go_next()

    def on_not_present():
        detections[unannotated_indices[index]]['human_feedback'][annotation_name] = 'False'
        go_next()

    def on_unclear():
        detections[unannotated_indices[index]]['human_feedback'][annotation_name] = 'Unclear'
        go_next()

    def on_construction():
        detections[unannotated_indices[index]]['human_feedback'][annotation_name] = 'Construction'
        go_next()
    
    def save_img():
        raw = root.current_raw_img
        ann = root.current_ann_img
        if raw is None or ann is None:
            messagebox.showwarning("Nothing to save", "No image is currently loaded.")
            return

        # combine them horizontally
        w, h = raw.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(raw, (0, 0))
        combined.paste(ann, (w, 0))

        # ask the user where to save
        default_name = f"{unannotated_indices[index]:03d}.png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialdir=path_export_example_images,
            initialfile=default_name
        )
        if not path:
            return

        try:
            combined.save(path, format="PNG")
            messagebox.showinfo("Saved", f"Image successfully saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error saving", str(e))
    
    #  Create buttons in the right column under the zoomed image
    btn_present = tk.Button(frame_left, text="Object", bg="green", fg="white", command=on_present)
    btn_not_present = tk.Button(frame_left, text="Background", bg="red", fg="white", command=on_not_present)
    btn_unclear = tk.Button(frame_left, text="Unclear", bg="orange", fg="white", command=on_unclear)
    btn_construction = tk.Button(frame_left, text="Construction", bg="yellow", fg="black", command=on_construction)
    btn_quit = tk.Button(frame_left, text="Quit", bg="gray", fg="white", command=quit_app)
    ptn_export = tk.Button(frame_left, text="Export", bg="blue", fg="white", command=save_img)  # NEW BUTTON

    btn_present.pack(fill=tk.X, pady=2)
    btn_not_present.pack(fill=tk.X, pady=2)
    tk.Label(frame_left, text="").pack(expand=True)
    btn_unclear.pack(fill=tk.X, pady=2)
    btn_construction.pack(fill=tk.X, pady=2)
    tk.Label(frame_left, text="").pack(expand=True)
    tk.Label(frame_left, text="").pack(expand=True)
    btn_quit.pack(fill=tk.X, pady=5)
    ptn_export.pack(fill=tk.X, pady=5)

    if len(unannotated_indices) > 0:
        show_image(unannotated_indices[index])
        root.mainloop()

    return detections


def annotate_image(
        image, 
        annotations, 
        x, 
        y, 
        window_size, 
        resolution,
        annotation_threshold
    ):

    config_style = {
        "tank":    {"color": (255, 69, 0) , "alpha": 0.5, "thickness": 1}, #, "transform": transform_tank_label
        "pile":    {"color": (30, 173, 255),   "alpha": 0.5, "thickness": 1}, #, "transform": transform_pile_label
        "all": {"color": (232, 23, 237),   "alpha": 0.5, "thickness": 1}, #, "transform": transform_all_label
    }

    categories = { # WARNING WRONG ????
        1: "tank",
        2: "pile",
        3: "all",
    }


    annotated_image = image.copy()

    def transform_segmentation(segmentation):
        """
        Convert a flat list of EPSG coordinates (e.g. [x1, y1, x2, y2, ...]) into pixel coordinates.
        """
        window_size_metre = window_size * resolution
        x_geo_min, y_geo_min, x_geo_max, y_geo_max = x - window_size_metre / 2, y - window_size_metre / 2, x + window_size_metre / 2, y + window_size_metre / 2
        x_pix_min, y_pix_min, x_pix_max, y_pix_max = 0, 0, window_size, window_size

        pix_width = x_pix_max - x_pix_min
        cart_width = x_geo_max - x_geo_min
        pix_height = y_pix_max - y_pix_min
        cart_height = y_geo_max - y_geo_min

        points = []
        for i in range(0, len(segmentation), 2):
            x_epsg = segmentation[i]
            y_epsg = segmentation[i+1]
            # Linear mapping for x
            x_pix = ((x_epsg - x_geo_min) / cart_width) * pix_width + x_pix_min
            y_pix = ((y_geo_max - y_epsg) / cart_height) * pix_height + y_pix_min
            points.append((int(x_pix), int(y_pix)))
        return np.array(points, np.int32)
    
    def annotate_single(annotation, img):
        cat_id = annotation.get("category_id")
        cat_name = categories.get(cat_id, "all")
        style = config_style[cat_name]
        color = style["color"]
        alpha = style["alpha"]
        thickness = style["thickness"]

        seg = annotation["segmentation"][0]
        pts = transform_segmentation(seg)
        # Process the segmentation: assume the first polygon in the list is used
        overlay = img.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        return img

    annotated_image = annotate_single(annotations, annotated_image)
    # Also annotate any nested (inside) annotations
    if "inside_list" in annotations:
        for inside_ann in annotations["inside_list"]:
            if inside_ann['score'] > annotation_threshold:
                annotated_image = annotate_single(inside_ann, annotated_image)

    # Convert the annotated image from BGR to RGB for PIL display
    #annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    #pil_ann_img = Image.fromarray(annotated_image_rgb)


    return annotated_image


def old_format_to_geojson(
        detections, 
        return_gdf = False
    ):
    features = []
    for detec in detections:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": detec['position']
            },
            "properties": detec
        }
        features.append(feature)

    if return_gdf:
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:2154")
        return gdf
    else:
        return features


def count_labels_per_source_old(detections):
    counts = {}

    for detec in detections:
        for source, val in detec['human_feedback'].items():

            if source not in counts:
                counts[source] = 0
            
            counts[source] += 1
    
    print('Counts per source:', counts)
    print('for: ', len(detections), 'detections')
                    
    return counts

