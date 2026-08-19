"""
Fine-tune DeepLabV3-ResNet50 on COCO panoptic's STUFF categories only.

Expects the standard COCO panoptic 2017 download layout:
    COCO_ROOT/
        annotations/
            panoptic_train2017.json
            panoptic_val2017.json
            panoptic_train2017/   <- folder of per-image PNGs (id-encoded RGB)
            panoptic_val2017/
        train2017/                <- raw jpgs
        val2017/

Install (same venv as everything else - pure PyTorch, no new heavy deps):
    pip install pillow numpy
    # torch, torchvision already present
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision import transforms as T

IGNORE_INDEX = 255


# --------------------------------------------------------------------------
# Step 1: derive the 53 stuff categories directly from the COCO panoptic JSON
# --------------------------------------------------------------------------
def build_stuff_mapping(panoptic_json_path):
    with open(panoptic_json_path) as f:
        data = json.load(f)

    stuff_cats = sorted(
        [c for c in data["categories"] if c["isthing"] == 0],
        key=lambda c: c["id"],
    )
    cat_id_to_train_id = {c["id"]: i for i, c in enumerate(stuff_cats)}
    train_id_to_name = {i: c["name"] for i, c in enumerate(stuff_cats)}

    print(f"Found {len(stuff_cats)} stuff categories (train ids 0..{len(stuff_cats)-1})")
    return cat_id_to_train_id, train_id_to_name, data["annotations"]


# --------------------------------------------------------------------------
# Step 2: dataset - decode panoptic PNG + segments_info -> semantic labels
# --------------------------------------------------------------------------
def decode_panoptic_png(png_path):
    """RGB-encoded segment id: id = R + G*256 + B*256^2"""
    rgb = np.array(Image.open(png_path).convert("RGB"), dtype=np.int32)
    return rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 256 * 256


class CocoStuffDataset(Dataset):
    def __init__(self, images_dir, panoptic_png_dir, annotations,
                 cat_id_to_train_id, image_size=512, augment=True):
        self.images_dir = images_dir
        self.panoptic_png_dir = panoptic_png_dir
        self.annotations = annotations   # list of {file_name, image_id, segments_info}
        self.cat_id_to_train_id = cat_id_to_train_id
        self.image_size = image_size
        self.augment = augment

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annotations)

    def _build_label(self, ann):
        png_path = os.path.join(self.panoptic_png_dir, ann["file_name"])
        id_map = decode_panoptic_png(png_path)

        # per-image remap: segment id -> train id (or ignore)
        max_id = int(id_map.max())
        lut = np.full(max_id + 1, IGNORE_INDEX, dtype=np.uint8)  # default: ignore (things/void)
        for seg in ann["segments_info"]:
            train_id = self.cat_id_to_train_id.get(seg["category_id"])
            if train_id is not None:            # only stuff categories get a real label
                lut[seg["id"]] = train_id
        # any pixel id not in segments_info (shouldn't happen) also falls to ignore via clip
        id_map_clipped = np.clip(id_map, 0, max_id)
        label = lut[id_map_clipped]
        return label

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.images_dir, ann["file_name"].replace(".png", ".jpg"))
        image = Image.open(img_path).convert("RGB")
        label = self._build_label(ann)
        label_img = Image.fromarray(label, mode="L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        label_img = label_img.resize((self.image_size, self.image_size), Image.NEAREST)  # NEAREST - never interpolate label ids

        image_t = self.normalize(image)
        label_t = torch.from_numpy(np.array(label_img, dtype=np.int64))
        return image_t, label_t


# --------------------------------------------------------------------------
# Step 3: swap the classifier head, keep backbone + ASPP as warm start
# --------------------------------------------------------------------------
def build_model(num_stuff_classes, device="cuda"):
    # aux_loss must be True here - DEFAULT weights were trained with an aux
    # classifier, so torchvision rejects aux_loss=False at load time.
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # we don't use the aux output in the training loop below, so disable it
    # after construction - saves the extra forward compute and extra output.
    model.aux_classifier = None

    # classifier is a Sequential; last element is the 1x1 conv (256 -> 21).
    # Replace ONLY that layer - backbone + ASPP keep their pretrained weights.
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_stuff_classes, kernel_size=1)

    return model.to(device)


# --------------------------------------------------------------------------
# Step 4: training loop
# --------------------------------------------------------------------------
def train(coco_root, epochs=10, batch_size=8, image_size=512, device="cuda",
          out_path="deeplabv3_coco_stuff.pt", val_split=0.1, seed=42):
    #
    # panoptic_json = os.path.join(coco_root, "annotations", "panoptic_train2017.json")
    # panoptic_png_dir = os.path.join(coco_root, "annotations", "panoptic_train2017")
    # images_dir = os.path.join(coco_root, "train2017")


    panoptic_json = os.path.join(coco_root, "annotations", "panoptic_val2017.json")
    panoptic_png_dir = os.path.join(coco_root, "annotations", "panoptic_val2017")
    images_dir = os.path.join(coco_root, "val2017")

    cat_id_to_train_id, train_id_to_name, annotations = build_stuff_mapping(panoptic_json)
    num_classes = len(train_id_to_name)
    # --- train/val split (default 90/10, set val_split=0.2 for 80/20) ---
    rng = random.Random(seed)
    shuffled = annotations[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_split))
    train_annotations = shuffled[:split_idx]
    val_annotations = shuffled[split_idx:]
    print(f"Split: {len(train_annotations)} train / {len(val_annotations)} val "
          f"({int((1 - val_split) * 100)}/{int(val_split * 100)})")

    train_dataset = CocoStuffDataset(images_dir, panoptic_png_dir, train_annotations,
                                     cat_id_to_train_id, image_size=image_size)
    val_dataset = CocoStuffDataset(images_dir, panoptic_png_dir, val_annotations,
                                   cat_id_to_train_id, image_size=image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    model = build_model(num_classes, device=device)
    model.train()

    # Collect child-layer params directly instead.
    head_children = list(model.classifier.children())
    pretrained_head_params = []
    for layer in head_children[:-1]:
        pretrained_head_params += list(layer.parameters())
    # lower LR on pretrained backbone/ASPP, higher LR on the fresh head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": pretrained_head_params, "lr": 1e-4},   # pretrained ASPP layers
        {"params": model.classifier[-1].parameters(), "lr": 1e-3},    # fresh 1x1 conv
    ])
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.cuda.amp.GradScaler()   # fp16 mixed precision - fits comfortably on 6GB
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(images)["out"]                    # (B, num_classes, H/8, W/8) internally, upsampled to input res
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} trainloss {loss.item():.4f}")

        avg_train_loss = total_loss/len(train_loader)

        print(f"epoch {epoch} avg loss {avg_train_loss:.4f}")

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    out = model(images)["out"]
                    loss = criterion(out, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"epoch {epoch}  train_loss {avg_train_loss:.4f}  val_loss {avg_val_loss:.4f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "train_id_to_name": train_id_to_name,
            "cat_id_to_train_id": cat_id_to_train_id,
            "num_classes": num_classes,
            "epoch": epoch,
            "val_loss": avg_val_loss,
        }

        torch.save(checkpoint, out_path)
        print(f"checkpoint saved -> {out_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = out_path.replace(".pt", "_best.pt")
            torch.save(checkpoint, best_path)
            print(f"new best val_loss {best_val_loss:.4f} -> {best_path}")

    return model, train_id_to_name


if __name__ == "__main__":
    train(
        coco_root="../../panoptic_annotations_trainval2017/",     # <-- point this at your downloaded COCO panoptic 2017 root
        epochs=10,
        batch_size=8,                  # reduce if you hit OOM on 6GB - try 4
        image_size=512,
        device="cuda",
        out_path="deeplabv3_coco_stuff.pt",
        val_split=0.1,
    )
