"""
Panoptic segmentation - YOLO (things) + DeepLabV3 (stuff), corrected fusion.

Fix from previous version: the semantic branch's THING-category predictions
(car, person, bus, etc.) are now discarded entirely. Only its STUFF-category
predictions are used to fill unclaimed pixels. This matters because:
  - torchvision's deeplabv3_resnet50 (VOC-21 weights) predicts thing classes too
  - unfiltered, those leaked into the panoptic map as instance-less "stuff cars"
  - that's invalid panoptic output and will corrupt PQ scoring against GT

IMPORTANT CAVEAT (flagging honestly, not hiding it):
torchvision's default VOC-21 deeplabv3_resnet50 has almost NO genuine stuff
classes (road, sky, building, grass, wall are not in VOC's 21 categories -
they're nearly all things: person, car, bus, dog, etc). After filtering to
true stuff, you'll get mostly "background" and very little else. This code
is correct regardless of which classes are stuff vs thing - swap
STUFF_CLASS_IDS to a Cityscapes/COCO-Stuff/ADE20K-pretrained model's mapping
whenever you're ready for richer stuff coverage. The fusion logic doesn't change.

Output format matches panopticapi (cocodataset/panopticapi) conventions:
    - one PNG per image, pixel RGB encodes segment id (id = R + G*256 + B*256^2)
    - one JSON with segments_info: [{id, category_id, area, bbox, iscrowd}, ...]
This is what pq_compute.py expects for both prediction and ground truth.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from ultralytics import YOLO


# --------------------------------------------------------------------------
# Category configuration - now sourced from the fine-tuned checkpoint's
# train_id_to_name (produced by finetune_deeplabv3_coco_stuff.py), instead
# of the old hardcoded VOC-21 list.
# --------------------------------------------------------------------------
STUFF_CHECKPOINT_PATH = "deeplabv3_coco_stuff_best.pt"


def load_stuff_metadata(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu",weights_only=True)
    train_id_to_name = ckpt["train_id_to_name"]     # {0: "banner", 1: "blanket", ...}
    num_classes = ckpt["num_classes"]
    return train_id_to_name, num_classes


# every train id from the fine-tuned model is genuine stuff now - no filtering needed
def build_category_table(yolo_names: dict, stuff_train_id_to_name: dict):
    categories = {}
    next_id = 1
    for _, name in sorted(yolo_names.items()):
        categories[name] = {"id": next_id, "name": name, "isthing": 1}
        next_id += 1
    for train_id, name in stuff_train_id_to_name.items():
        categories[f"stuff_{name}"] = {"id": next_id, "name": name, "isthing": 0, "train_id": train_id}
        next_id += 1
    return categories


# --------------------------------------------------------------------------
# "Things" - instance segmentation
# --------------------------------------------------------------------------
class InstanceSeg:
    def __init__(self, weights: str = "yolo11n-seg.pt", device: str = "cuda", conf: float = 0.4):
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.names = self.model.names   # {id: name}

    def __call__(self, image_bgr: np.ndarray):
        results = self.model.predict(image_bgr, device=self.device, conf=self.conf, verbose=False)[0]
        instances = []
        if results.masks is None:
            return instances

        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes
        h, w = image_bgr.shape[:2]

        for i in range(len(masks)):
            mask_t = torch.from_numpy(masks[i])[None, None].float()
            mask_resized = F.interpolate(mask_t, size=(h, w), mode="bilinear", align_corners=False)
            mask_bool = (mask_resized.squeeze().numpy() > 0.5)

            instances.append({
                "mask": mask_bool,
                "class_id": int(boxes.cls[i].item()),
                "class_name": self.names[int(boxes.cls[i].item())],
                "score": float(boxes.conf[i].item()),
            })

        instances.sort(key=lambda x: x["score"], reverse=True)
        return instances


# --------------------------------------------------------------------------
# "Stuff" - semantic segmentation, fine-tuned on COCO panoptic stuff classes
# --------------------------------------------------------------------------
class SemanticSeg:
    def __init__(self, checkpoint_path: str = STUFF_CHECKPOINT_PATH, device: str = "cuda"):
        self.train_id_to_name, num_classes = load_stuff_metadata(checkpoint_path)

        # rebuild the same architecture used in fine-tuning, then load your weights
        self.model = deeplabv3_resnet50(weights=None,weights_backbone=None,aux_loss=False)
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()

        self.device = device
        # match the normalization used during fine-tuning (ImageNet stats)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def __call__(self, image_pil: Image.Image):
        img_t = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(self.device)
        img_t = (img_t - self._mean) / self._std

        out = self.model(img_t)["out"]
        out = F.interpolate(out, size=image_pil.size[::-1], mode="bilinear", align_corners=False)
        return out.argmax(dim=1).squeeze(0).cpu().numpy()   # values are now train_ids 0..num_classes-1, all genuine stuff


# --------------------------------------------------------------------------
# Fusion - CORRECTED: only genuine stuff classes fill unclaimed pixels
# --------------------------------------------------------------------------
def fuse_panoptic(instances, semantic_map, image_hw, categories, stuff_train_id_to_name,
                   instance_overlap_thresh: float = 0.5,
                   min_stuff_area: int = 500):
    h, w = image_hw
    panoptic_map = np.zeros((h, w), dtype=np.int32)
    claimed = np.zeros((h, w), dtype=bool)
    segments_info = []
    next_id = 1

    # 1) things - highest confidence first
    for inst in instances:
        mask = inst["mask"]
        overlap = (mask & claimed).sum() / max(mask.sum(), 1)
        if overlap > instance_overlap_thresh:
            continue

        new_pixels = mask & ~claimed
        if new_pixels.sum() < 50:
            continue

        ys, xs = np.where(new_pixels)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]

        panoptic_map[new_pixels] = next_id
        claimed |= new_pixels
        segments_info.append({
            "id": next_id,
            "category_id": categories[inst["class_name"]]["id"],
            "area": int(new_pixels.sum()),
            "bbox": bbox,
            "iscrowd": 0,
        })
        next_id += 1

    # 2) stuff - every train_id from the fine-tuned model is genuine stuff,
    # no filtering needed (unlike the old VOC-21 model, which mixed thing+stuff)
    remaining = ~claimed
    for train_id, name in stuff_train_id_to_name.items():
        cls_mask = remaining & (semantic_map == train_id)
        area = cls_mask.sum()
        if area < min_stuff_area:
            continue

        ys, xs = np.where(cls_mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]

        panoptic_map[cls_mask] = next_id
        segments_info.append({
            "id": next_id,
            "category_id": categories[f"stuff_{name}"]["id"],
            "area": int(area),
            "bbox": bbox,
            "iscrowd": 1,
        })
        next_id += 1

    # any pixel not claimed by a valid thing or stuff segment stays VOID (id=0),
    # exactly as panopticapi expects - never silently mislabeled.
    return panoptic_map, segments_info


# --------------------------------------------------------------------------
# COCO-panoptic-format output: PNG (id-encoded RGB) + JSON
# --------------------------------------------------------------------------
def panoptic_map_to_png(panoptic_map: np.ndarray) -> Image.Image:
    h, w = panoptic_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = panoptic_map % 256
    rgb[..., 1] = (panoptic_map // 256) % 256
    rgb[..., 2] = (panoptic_map // (256 * 256)) % 256
    return Image.fromarray(rgb, mode="RGB")


def save_panoptic_result(panoptic_map, segments_info, image_id, out_png_path, out_json_path):
    png = panoptic_map_to_png(panoptic_map)
    png.save(out_png_path)

    record = {
        "image_id": image_id,
        "file_name": out_png_path.split("/")[-1],
        "segments_info": segments_info,
    }
    with open(out_json_path, "w") as f:
        json.dump(record, f, indent=2)


# --------------------------------------------------------------------------
# Loosely coupled depth plugin interface - implement later with DepthAnythingV2
# --------------------------------------------------------------------------
class DepthEstimator:
    """
    Abstract interface. Swap in a DepthAnythingV2 implementation later without
    touching InstanceSeg, SemanticSeg, or fuse_panoptic - the panoptic branch
    has zero dependency on this class existing at all.
    """
    def __call__(self, image_pil: Image.Image) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Full panoptic wrapper - plain YOLO + DeepLabV3 integration, no depth yet
# --------------------------------------------------------------------------
class PanopticPipeline:
    def __init__(self, device: str = "cuda", yolo_weights: str = "yolo11s-seg.pt",
                 stuff_checkpoint: str = STUFF_CHECKPOINT_PATH):
        self.instance_seg = InstanceSeg(weights=yolo_weights, device=device)
        self.semantic_seg = SemanticSeg(checkpoint_path=stuff_checkpoint, device=device)
        self.categories = build_category_table(self.instance_seg.names, self.semantic_seg.train_id_to_name)

    def __call__(self, image_path: str):
        image_pil = Image.open(image_path).convert("RGB")
        image_bgr = np.array(image_pil)[:, :, ::-1]

        instances = self.instance_seg(image_bgr)
        semantic_map = self.semantic_seg(image_pil)

        h, w = image_pil.size[1], image_pil.size[0]
        panoptic_map, segments_info = fuse_panoptic(
            instances, semantic_map, (h, w), self.categories, self.semantic_seg.train_id_to_name
        )

        return {
            "panoptic_map": panoptic_map,
            "segments_info": segments_info,
            "categories": self.categories,
        }



# --------------------------------------------------------------------------
# Debug printing - resolve numeric category_id back to a readable name.
# JSON output stays numeric (COCO-panoptic-format compliant); this is
# display-only.
# --------------------------------------------------------------------------
def segments_info_with_names(segments_info, categories):
    id_to_name = {v["id"]: v["name"] for v in categories.values()}
    resolved = []
    for seg in segments_info:
        seg = dict(seg)
        seg["name"] = id_to_name.get(seg["category_id"], f"unknown({seg['category_id']})")
        resolved.append(seg)
    return resolved


# --------------------------------------------------------------------------
# Visualization - overlay instance masks (things) + semantic masks (stuff),
# each with its resolved class name drawn on top.
# --------------------------------------------------------------------------
import cv2

def visualize_panoptic(image_bgr, panoptic_map, segments_info, categories, alpha=0.5):
    id_to_name = {v["id"]: v["name"] for v in categories.values()}
    vis = image_bgr.copy()
    rng = np.random.default_rng(42)  # fixed seed -> stable colors across frames

    for seg in segments_info:
        seg_mask = (panoptic_map == seg["id"])
        if seg_mask.sum() == 0:
            continue

        color = rng.integers(60, 255, size=3).tolist()
        overlay = vis.copy()
        overlay[seg_mask] = color
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        ys, xs = np.where(seg_mask)
        label_x, label_y = int(xs.min()), int(ys.min()) - 6
        name = id_to_name.get(seg["category_id"], "?")
        kind = "thing" if any(c["id"] == seg["category_id"] and c["isthing"] == 1
                               for c in categories.values()) else "stuff"
        text = f"{name} ({kind})"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (label_x, max(0, label_y - th - 4)),
                      (label_x + tw + 4, max(0, label_y)), (0, 0, 0), -1)
        cv2.putText(vis, text, (label_x + 2, max(12, label_y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return vis

import os
if __name__ == "__main__":
    pipeline = PanopticPipeline(device="cuda", yolo_weights="yolo11m-seg.pt")
    fldr_pth = os.path.join(os.curdir,"test")
    fldr_lst = os.listdir(fldr_pth)
    for fl in fldr_lst:
        img_pth = os.path.join(fldr_pth,fl)
        out = pipeline(img_pth)
        lst_thngs=[]
        lst_stuff=[]
        named = segments_info_with_names(out["segments_info"], out["categories"])
        # print(f"Segments: {len(named)}")
        for s in named[:8]:
            # print(s)   # now includes readable "name" field alongside category_id
            print(s["name"])

            if s["iscrowd"] == 1:
                lst_thngs.append(s["name"])
            elif s["iscrowd"] == 0:
                lst_stuff.append(s["name"])

        save_panoptic_result(
            out["panoptic_map"], out["segments_info"],
            image_id=fl,
            out_png_path=fl+".png",
            out_json_path=fl+".json",
        )
        # print("Saved sample_panoptic.png + sample_panoptic.json (panopticapi-compatible)")

        image_bgr = cv2.imread(img_pth)
        vis = visualize_panoptic(image_bgr, out["panoptic_map"], out["segments_info"], out["categories"])
        fl_vis = fl.split(".")[0]+"vis.jpg"
        msk_img_pth = os.path.join(os.curdir,fl_vis)
        cv2.imwrite(msk_img_pth, vis)
        # print("Saved sample_panoptic_visualized.jpg (masks + class names)")
