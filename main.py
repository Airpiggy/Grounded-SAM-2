import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy

# This demo shows the continuous object tracking plus reverse tracking with Grounding DINO and SAM 2
"""
Step 1: Environment settings and model initialization
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from local checkpoint
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=device
)

print(f"Grounding DINO model loaded from local checkpoint: {GROUNDING_DINO_CHECKPOINT}")


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "table. chair. robot. shelf. floor."

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
video_dir = "assets/extracted_data/color_jpg/"
# 'output_dir' is the directory to save the annotated frames
output_dir = "outputs"
# 'output_video_path' is the path to save the final video
output_video_path = "./outputs/output.mp4"
# create the output directory
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)
step = 20 # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0
frame_object_count = {}
"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", len(frame_names))
for start_frame_idx in range(0, len(frame_names), step):
# prompt grounding dino to get the box coordinates on specific frame
    print("start_frame_idx", start_frame_idx)
    video_segments = {}  # output the following {step} frames tracking masks
    # continue
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    
    # Load image using Grounding DINO's load_image function
    image_source, image = load_image(img_path)
    
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

    # run Grounding DINO on the image using local model (WITHOUT bfloat16)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box coordinates for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes.to(device) * torch.tensor([w, h, w, h], device=device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    
    # Convert to numpy for SAM2
    input_boxes_np = input_boxes.cpu().numpy()
    confidences_np = confidences.cpu().numpy().tolist()
    
    # process the detection results
    OBJECTS = labels
    
    # NOW enable bfloat16 for SAM2 (like official script)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(image_source)
        
        if input_boxes.shape[0] != 0:
            print(f"Detected {len(OBJECTS)} objects: {OBJECTS}")

            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes_np,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)
        else:
            masks = None
    
    """
    Step 3: Register each object's positive points to video predictor
    """
    if masks is not None and input_boxes.shape[0] != 0:
        # If you are using point prompts, we uniformly sample positive points based on the mask
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=input_boxes, label_list=OBJECTS)
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")
    else:
        print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
        mask_dict = sam2_masks

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
    frame_object_count[start_frame_idx] = objects_count
    print("objects_count", objects_count)
    
    if len(mask_dict.labels) == 0:
        mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
        print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
        continue
    else:
        video_predictor.reset_state(inference_state)

        # Use bfloat16 for SAM 2 video tracking
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for object_id, object_info in mask_dict.labels.items():
                video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )
            
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id), logit=mask_dict.get_target_logit(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

            print("video_segments:", len(video_segments))
    """
    Step 5: save the tracking masks and json files
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        frame_masks_info.to_json(json_data_path)
       

CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

print("try reverse tracking")
start_object_id = 0
object_info_dict = {}

# Use bfloat16 for reverse tracking
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    for frame_idx, current_object_count in frame_object_count.items():
        print("reverse tracking frame", frame_idx, frame_names[frame_idx])

        if frame_idx == 0:
            start_object_id = current_object_count
            print(f"Skip reverse tracking for first frame")
            continue
         
        video_predictor.reset_state(inference_state)
        image_base_name = frame_names[frame_idx].split(".")[0]
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        json_data = MaskDictionaryModel().from_json(json_data_path)
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
        mask_array = np.load(mask_data_path)
        
        has_new_objects_id = False
        for object_id in range(start_object_id+1, current_object_count+1):
            print("reverse tracking object", object_id)
            object_info_dict[object_id] = json_data.labels[object_id]
            mask_tensor = torch.from_numpy((mask_array == object_id).astype(np.uint8)).to(device)
            video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_tensor)
            has_new_objects_id = True
        start_object_id = current_object_count

        if not has_new_objects_id:
            print(f"Skip frame {frame_idx} - no new objects")
            continue
        
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state, max_frame_num_to_track=step*2, start_frame_idx=frame_idx, reverse=True
        ):
            image_base_name = frame_names[out_frame_idx].split(".")[0]
            json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
            json_data = MaskDictionaryModel().from_json(json_data_path)
            mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
            mask_array = np.load(mask_data_path)
            # merge the reverse tracking masks with the original masks
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu()
                if out_mask.sum() == 0:
                    print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                    continue
                object_info = object_info_dict[out_obj_id]
                object_info.mask = out_mask[0]
                object_info.update_box()
                json_data.labels[out_obj_id] = object_info
                mask_cpu = object_info.mask.numpy()
                mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                mask_array[mask_cpu] = out_obj_id  
                        
            np.save(mask_data_path, mask_array)
            json_data.to_json(json_data_path)


"""
Step 6: Draw the results and save the video
"""
CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir+"_reverse")

create_video_from_images(result_dir+"_reverse", output_video_path, frame_rate=10)