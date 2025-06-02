
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class Segment()
    
    def __init__(self):
        device = 'cuda'
        output_mode = "binary_mask"

        #checkpoint_path = BASE + "weights/SAM/sam_vit_b_01ec64.pth"
        #model_type = 'vit_b'
        checkpoint_path = BASE + "weights/SAM/sam_vit_l_0b3195.pth"
        model_type = 'vit_l'

        print(sam_model_registry)


        sam_kwargs = {
            "points_per_side": 64,                    # default: 32
            "points_per_batch": 64,                    # default: 64
            "pred_iou_thresh": 0.6,                    # default: 0.88
            "stability_score_thresh": 0.5,             # default: 0.95
            "stability_score_offset": 1,               # default: 1
            "box_nms_thresh": 0.7,                     # default: 0.7
            "crop_n_layers": 0,                        # default: 0
            "crop_nms_thresh": 0.7,                    # default: 0.7
            "crop_overlap_ratio": 512 / 1500,          # default: 512/1500 ≈ 0.341
            "crop_n_points_downscale_factor": 1,       # default: 1
            "min_mask_region_area": 70000,               # default: 0 # not working 
        }


        sam_kwargs = {k: v for k, v in sam_kwargs.items() if v is not None}
        sam_kwargs
        print("Loading model...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        _ = sam.to(device=device)
        self.generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **sam_kwargs)
    
    def __call__(self, image):
        image = cv2.imread(image_file) # rgb ??
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        masks = generator.generate(image)
        
        print('Masks befor filtering',len(masks))

        def get_mask_compactness(binary_mask):
            mask_uint8 = binary_mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = np.count_nonzero(mask_uint8)
            perimeter = sum(cv2.arcLength(cnt, closed=True) for cnt in contours)
            if perimeter == 0:
                return 0.0
            compactness = area / (perimeter ** 2)
            return compactness
        
        
        min_area_pixel = 500
        min_score = 0.5
        min_shape_factor = 0.01
        threshold = 0.3
        place_mask = segmentation_to_mask(annotation['pixels']['segm'][0], image.shape)

        # Filter masks by IoU
        masks = [m for m in masks if compute_overlap(m['segmentation'], place_mask) > threshold]
        masks = [m for m in masks if m['area'] > min_area_pixel]
        masks = [m for m in masks if m['predicted_iou'] > min_score]

        for mask in masks:
            mask["predicted_iou"] = mask["predicted_iou"]* (1- mask['area']/width*height)
        

        image = np.array(image)
        label_map = np.zeros((height, width), dtype=np.int32)
        score_map = np.full((height, width), -np.inf)  # Track best score per pixel

        # Assign best mask id to each pixel
        for idx, mask in enumerate(masks, start=1):  # 1-based indexing
            segmentation = mask["segmentation"]
            score = mask["predicted_iou"] # 'stability_score'
            # Apply mask where this score is better
            update_pixels = (segmentation) & (score > score_map)


            label_map[update_pixels] = idx
            score_map[update_pixels] = score
        

        label_map[place_mask == 0] = 0
        
        for idx, _ in enumerate(masks, start=1):
            mask = label_map == idx
            shape_factor = get_mask_compactness(mask)
            if shape_factor < min_shape_factor:
                label_map[mask] = 0

        
        base_name = splitext(basename(image_file))[0]
        output_dir = join(path_to_masks, base_name)
        os.makedirs(output_dir, exist_ok=True)

        mask_file = join(output_dir, 'mask.npy')
        show_segmentation_file = join(output_dir, 'show.png')

        # Save label map
        np.save(mask_file, label_map)

        show_masks_2(
            image, 
            label_map, 
            mask_alpha = 0.4,
            verbose=False,
            save_path = show_segmentation_file
        )