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

CHANGELOG (fixes for val-loss divergence after epoch ~4):
    1. Train on the real train2017 split (118k imgs) / validate on val2017,
       instead of training on an 80% slice of val2017 (~4k imgs). The old
       setup starved the model of data and caused fast overfitting.
    2. `augment=True` now actually does something: random-resized-crop,
       horizontal flip, mild color jitter on the training split only.
    3. Added a cosine LR schedule so LR decays as training progresses,
       instead of holding a high head LR for all 40 epochs.
    4. Added early stopping (patience-based) so training stops once val
       loss has stopped improving instead of running the full 40 epochs.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision import transforms as T
import csv

IGNORE_INDEX = 255


# --------------------------------------------------------------------------
# Step 1: derive the 53 stuff categories directly from the COCO panoptic JSON
# --------------------------------------------------------------------------
def build_stuff_mapping(panoptic_json_path,panoptic_png_path):
    with open(panoptic_json_path) as f:
        data = json.load(f)
    train_png_lst = os.listdir(panoptic_png_path)
    stuff_cats = sorted(
        [c for c in data["categories"] if c["isthing"] == 0],
        key=lambda c: c["id"],
    )
    cat_id_to_train_id = {c["id"]: i for i, c in enumerate(stuff_cats)}
    train_id_to_name = {i: c["name"] for i, c in enumerate(stuff_cats)}

    print(f"Found {len(stuff_cats)} stuff categories (train ids 0..{len(stuff_cats)-1})")
    return cat_id_to_train_id, train_id_to_name, data["annotations"],train_png_lst


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
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

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
        # if self.train:
        #     ann = self.
        img_path = os.path.join(self.images_dir, ann["file_name"].replace(".png", ".jpg"))
        image = Image.open(img_path).convert("RGB")
        label = self._build_label(ann)
        label_img = Image.fromarray(label, mode="L")

        if self.augment:
            # random-resized-crop: same crop box applied to image + label
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.5, 1.0), ratio=(0.9, 1.1))
            image = T.functional.resized_crop(
                image, i, j, h, w, (self.image_size, self.image_size), Image.BILINEAR)
            label_img = T.functional.resized_crop(
                label_img, i, j, h, w, (self.image_size, self.image_size), Image.NEAREST)

            if random.random() < 0.5:
                image = T.functional.hflip(image)
                label_img = T.functional.hflip(label_img)

            # color jitter on the image only - never touches label ids
            image = self.color_jitter(image)
        else:
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
def train(coco_root, epochs=40, batch_size=8, image_size=512, device="cuda",
          out_path="deeplabv3_coco_stuff.pt", log_csv_path="coco_52ktraining_log.csv",
          seed=42, patience=5, head_weight_decay=0.05):

    torch.backends.cudnn.benchmark = True # code change 2

    train_panoptic_json = os.path.join(coco_root, "annotations", "panoptic_train2017.json")
    train_panoptic_png_dir = os.path.join(coco_root, "annotations", "panoptic_train2017")
    train_images_dir = os.path.join(coco_root, "train2017")

    val_panoptic_json = os.path.join(coco_root, "annotations", "panoptic_val2017.json")
    val_panoptic_png_dir = os.path.join(coco_root, "annotations", "panoptic_val2017")
    val_images_dir = os.path.join(coco_root, "val2017")

    cat_id_to_train_id, train_id_to_name, train_annotations, train_segment_lst = build_stuff_mapping(train_panoptic_json,train_panoptic_png_dir)
    num_classes = len(train_id_to_name)

    # shuffle train annotations for good measure (DataLoader also shuffles, but
    # keeps this deterministic/seeded if you ever want reproducible ordering)
    rng = random.Random(seed)

    trn_segm_set = set(train_segment_lst)
    train_annotations[:] = [ann for ann in train_annotations if ann["file_name"] in trn_segm_set]
    rng.shuffle(train_annotations)

    val_png_lst = os.listdir(val_panoptic_png_dir)
    val_segm_set = set(val_png_lst)
    with open(val_panoptic_json) as f:
        val_annotations = json.load(f)["annotations"]
    val_annotations[:] = [ann for ann in val_annotations if ann["file_name"] in val_segm_set]

    print(f"Train: {len(train_annotations)} images (train2017) / "
          f"Val: {len(val_annotations)} images (val2017)")

    train_dataset = CocoStuffDataset(train_images_dir, train_panoptic_png_dir, train_annotations,
                                      cat_id_to_train_id, image_size=image_size, augment=True)
    val_dataset = CocoStuffDataset(val_images_dir, val_panoptic_png_dir, val_annotations,
                                    cat_id_to_train_id, image_size=image_size, augment=False)
    # Separate the batch sizes to maximize your 8GB GPU potential
    # TRAIN_BATCH_SIZE = 16
    # VAL_BATCH_SIZE = 32  # Validation handles larger batches easily since it tracks zero gradients
#code change 3 persistent_workers=True added
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)#, persistent_workers=True)

    model = build_model(num_classes, device=device)
    # model = model.to(memory_format=torch.channels_last) #-- need to apply this changes
    # model = torch.compile(model)
    model.train()

    # Collect child-layer params directly instead.
    head_children = list(model.classifier.children())
    pretrained_head_params = []
    for layer in head_children[:-1]:
        pretrained_head_params += list(layer.parameters())

    # lower LR on pretrained backbone/ASPP, higher LR on the fresh head.
    # extra weight_decay on the fresh head only - it's the part most prone
    # to overfitting since it starts randomly initialized.
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": pretrained_head_params, "lr": 1e-4},                     # pretrained ASPP layers
        {"params": model.classifier[-1].parameters(), "lr": 1e-4,
         "weight_decay": head_weight_decay},                                # fresh 1x1 conv
    ])

    # cosine decay over the full run - keeps early LR high enough to adapt
    # the head, then tapers off instead of hammering at a fixed high LR
    # for all 40 epochs (a big driver of the late-epoch overfitting).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler('cuda')   # fp16 mixed precision - fits comfortably on 6GB
    best_val_loss = float("inf")
    patience_counter = 0

    log_exists = os.path.exists(log_csv_path)
    log_file = open(log_csv_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["epoch", "train_loss", "val_loss", "lr_head"])
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for step, (images, labels) in enumerate(train_loader):
            data_time = time.time() - start_time
            # images, labels = images.to(device), labels.to(device)
            images = images.to(device, non_blocking=True) # need to change code
            labels = labels.to(device, non_blocking=True)
            print(f"After loading images of step {time.time()}")

            optimizer.zero_grad(set_to_none=True)#changed
            with torch.amp.autocast('cuda'):
                out = model(images)["out"]                    # (B, num_classes, H, W)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach() #total_loss += loss.item() #code change 1
            step_time = time.time() - start_time
            if step % 10 == 0:
                print(f"Batch {step}-> Data fetch:{data_time:.4f}s | Total Step: {step_time:.4f}s")
            start_time = time.time()
            #del images, labels, out, loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"epoch {epoch} avg loss {avg_train_loss:.4f}")

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(images)["out"]
                    loss = criterion(out, labels)
                val_loss += loss.item()
                del images, labels, out, loss
        torch.cuda.empty_cache()
        avg_val_loss = val_loss / max(len(val_loader), 1)
        current_head_lr = optimizer.param_groups[-1]["lr"]
        print(f"epoch {epoch}  train_loss {avg_train_loss:.4f}  val_loss {avg_val_loss:.4f}  "
              f"head_lr {current_head_lr:.6f}")
        log_writer.writerow([epoch, avg_train_loss, avg_val_loss, current_head_lr])
        log_file.flush()

        scheduler.step()

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
            patience_counter = 0
            best_path = out_path.replace(".pt", "_val_best.pt")
            torch.save(checkpoint, best_path)
            print(f"new best val_loss {best_val_loss:.4f} -> {best_path}")
        else:
            patience_counter += 1
            print(f"no val improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"early stopping at epoch {epoch} - best val_loss {best_val_loss:.4f}")
                break

    log_file.close()
    return model, train_id_to_name


if __name__ == "__main__":
    train(
        coco_root="../panoptic_annotations_trainval2017/",     # <-- point this at your downloaded COCO panoptic 2017 root
        epochs=10,
        batch_size=8,                  # reduce if you hit OOM on 6GB - try 4
        image_size=512,
        device="cuda",
        out_path="deeplabv3_coco_52ktrn_20epch_stuff.pt",
        patience=5,
    )
