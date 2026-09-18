"""
Incremental Semantic Segmentation:
COCO-Stuff 53 classes -> IDD-expanded 62 classes

Teacher:
    Original COCO-Stuff DeepLabV3-ResNet50
    53 classes
    FROZEN

Student:
    DeepLabV3-ResNet50
    62 classes
    First 53 classifier weights copied from COCO teacher
    9 new classifier channels randomly initialized

IDD classes:
    7 are mapped to existing COCO classes
    9 become new classes
    fallback background -> IGNORE (255)

Loss:
    L_total = L_CE + lambda_kd * L_KD

CE:
    Applied to valid IDD labels

KD:
    Teacher 53 logits vs Student first 53 logits
    Applied mainly on pixels where IDD provides no direct supervision
    and teacher confidence is sufficiently high.

Output:
    deeplabv3_coco_idd_62_best.pt
"""

import os
import json
import csv
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms as T


# ============================================================
# CONFIGURATION
# ============================================================

IGNORE_INDEX = 255

OLD_CLASSES = 53
NEW_CLASSES = 10
TOTAL_CLASSES = 63

KD_TEMPERATURE = 2.0
KD_LAMBDA = 1.0
KD_CONFIDENCE = 0.60

IMAGE_SIZE = 512

# ------------------------------------------------------------
# Final 62-class ontology
# ------------------------------------------------------------

FINAL_CLASSES = {
    0: "banner",
    1: "blanket",
    2: "bridge",
    3: "cardboard",
    4: "counter",
    5: "curtain",
    6: "door-stuff",
    7: "floor-wood",
    8: "flower",
    9: "fruit",
    10: "gravel",
    11: "house",
    12: "light",
    13: "mirror-stuff",
    14: "net",
    15: "pillow",
    16: "platform",
    17: "playingfield",
    18: "railroad",
    19: "river",
    20: "road",
    21: "roof",
    22: "sand",
    23: "sea",
    24: "shelf",
    25: "snow",
    26: "stairs",
    27: "tent",
    28: "towel",
    29: "wall-brick",
    30: "wall-stone",
    31: "wall-tile",
    32: "wall-wood",
    33: "water-other",
    34: "window-blind",
    35: "window-other",
    36: "tree-merged",
    37: "fence-merged",
    38: "ceiling-merged",
    39: "sky-other-merged",
    40: "cabinet-merged",
    41: "table-merged",
    42: "floor-other-merged",
    43: "pavement-merged",
    44: "mountain-merged",
    45: "grass-merged",
    46: "dirt-merged",
    47: "paper-merged",
    48: "food-other-merged",
    49: "building-other-merged",
    50: "rock-merged",
    51: "wall-other-merged",
    52: "rug-merged",

    # New IDD classes
    53: "billboard",
    54: "curb",
    55: "guard rail",
    56: "sidewalk",
    57: "parking",
    58: "drivable fallback",
    59: "non-drivable fallback",
    60: "obs-str-bar-fallback",
    61: "vegetation",
    62: "ground",
}


# ============================================================
# IDD -> FINAL GLOBAL CLASS ID
# ============================================================

IDD_TO_GLOBAL = {

    # Existing COCO classes
    "bridge": 2,
    "tunnel": 2,
    "building": 49,
    "fence": 37,
    "rail track": 18,
    "road": 20,
    "sky": 39,
    "wall": 51,

    # New IDD classes
    "billboard": 53,
    "curb": 54,
    "guard rail": 55,
    "sidewalk": 56,
    "parking": 57,
    "drivable fallback": 58,
    "non-drivable fallback": 59,
    "obs-str-bar-fallback": 60,
    "vegetation": 61,
    "ground": 62,
    # Not a trainable semantic class
    "fallback background": IGNORE_INDEX,
}

STUFF_LABELS = set(IDD_TO_GLOBAL.keys())
THING_LABELS = {
    "person",
    "rider",
    "motorcycle",
    "bicycle",
    "autorickshaw",
    "car",
    "truck",
    "bus",
    "vehicle fallback",
    "train",
    "trailer",
    "caravan",
    "animal",
    "traffic light",
    "traffic sign",
    "polegroup",
    "pole",
}

IGNORE_LABELS = {
    "unlabeled",
    "out of roi",
    "ego vehicle",
    "rectification border",
    "license plate",
}

def print_final_mapping():

    print("\n" + "=" * 70)
    print("FINAL 62-CLASS ONTOLOGY")
    print("=" * 70)

    for idx in range(TOTAL_CLASSES):
        print(f"{idx:02d}: {FINAL_CLASSES[idx]}")

    print("\nIDD -> GLOBAL ID")
    print("-" * 70)

    for name, gid in IDD_TO_GLOBAL.items():
        if gid == IGNORE_INDEX:
            print(f"{name:30s} -> IGNORE ({IGNORE_INDEX})")
        else:
            print(f"{name:30s} -> {gid:02d} ({FINAL_CLASSES[gid]})")

    print("=" * 70)


# ============================================================
# SCAN IDD LABELS
# ============================================================

def scan_idd_labels(json_dir):

    seen_labels = set()

    for fname in os.listdir(json_dir):

        if not fname.endswith(".json"):
            continue

        path = os.path.join(json_dir, fname)

        with open(path, "r") as f:
            data = json.load(f)

        for obj in data.get("objects", []):
            seen_labels.add(obj["label"])

    unknown = (
        seen_labels
        - STUFF_LABELS
        - THING_LABELS
        - IGNORE_LABELS
    )

    if unknown:
        raise ValueError(
            "\nUnknown IDD labels found:\n"
            f"{sorted(unknown)}\n\n"
            "Add these labels to STUFF_LABELS, THING_LABELS "
            "or IGNORE_LABELS before training."
        )

    print("\nIDD labels found:")
    for label in sorted(seen_labels):
        if label in STUFF_LABELS:
            print(f"  STUFF : {label:30s} -> {IDD_TO_GLOBAL[label]}")
        elif label in THING_LABELS:
            print(f"  THING : {label}")
        else:
            print(f"  IGNORE: {label}")

    return seen_labels


# ============================================================
# DATASET
# ============================================================

class IDDIncrementalDataset(Dataset):

    def __init__(
        self,
        images_dir,
        json_dir,
        file_stems,
        image_size=512,
        img_ext=".jpg"
    ):

        self.images_dir = images_dir
        self.json_dir = json_dir
        self.file_stems = file_stems
        self.image_size = image_size
        self.img_ext = img_ext

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.file_stems)

    # --------------------------------------------------------
    # Find image robustly
    # --------------------------------------------------------

    def find_image(self, stem):

        candidates = [

            os.path.join(
                self.images_dir,
                stem.split("_")[0] + "_leftImg8bit" + self.img_ext
            ),

            os.path.join(
                self.images_dir,
                stem + self.img_ext
            ),

        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        # Last resort: search by prefix
        prefix = stem.split("_")[0]

        for fname in os.listdir(self.images_dir):

            if fname.startswith(prefix):
                path = os.path.join(self.images_dir, fname)

                if os.path.isfile(path):
                    return path

        raise FileNotFoundError(
            f"Image not found for JSON stem: {stem}"
        )

    # --------------------------------------------------------
    # Build global 62-class label image
    # --------------------------------------------------------

    def build_label(self, json_path, width, height):

        # Everything initially ignored
        label_img = Image.new(
            "L",
            (width, height),
            color=IGNORE_INDEX
        )

        draw = ImageDraw.Draw(label_img)

        with open(json_path, "r") as f:
            data = json.load(f)

        for obj in data.get("objects", []):

            label = obj["label"]

            if label in IGNORE_LABELS:
                continue
            polygon = obj.get("polygon", [])
            if len(polygon) < 3:
                continue
            polygon = [tuple(map(int, pt)) for pt in polygon]
            if label in STUFF_LABELS:
                global_id = IDD_TO_GLOBAL[label]
                # fallback background = ignore
                if global_id == IGNORE_INDEX:
                    fill = IGNORE_INDEX
                else:
                    fill = global_id

            elif label in THING_LABELS:
                fill = IGNORE_INDEX
            else:
                raise ValueError(f"Unhandled IDD label: {label}")
            draw.polygon(polygon,fill=fill)
        return label_img

    def __getitem__(self, idx):
        stem = self.file_stems[idx]
        image_path = self.find_image(stem)
        json_path = os.path.join( self.json_dir,stem + ".json")
        image = Image.open(image_path).convert("RGB")
        label_img = self.build_label(json_path, image.width, image.height)
        image = image.resize( (self.image_size, self.image_size),Image.BILINEAR )

        label_img = label_img.resize( (self.image_size, self.image_size), Image.NEAREST )
        image_t = self.normalize(image)
        label_t = torch.from_numpy(np.array(label_img,dtype=np.int64) )
        return image_t, label_t

def build_teacher(    coco_checkpoint_path,   device):
    print("\nLoading COCO 53-class teacher...")
    checkpoint = torch.load( coco_checkpoint_path, map_location="cpu",weights_only=True    )
    coco_num_classes = checkpoint["num_classes"]
    if coco_num_classes != OLD_CLASSES:
        raise ValueError(   f"Expected COCO checkpoint with {OLD_CLASSES} classes, but found {coco_num_classes}")

    teacher = deeplabv3_resnet50(  weights=None,weights_backbone=None,aux_loss=False)
    in_channels = teacher.classifier[-1].in_channels
    teacher.classifier[-1] = nn.Conv2d( in_channels, OLD_CLASSES, kernel_size=1)

    teacher.load_state_dict( checkpoint["model_state_dict"])
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded: {coco_checkpoint_path}")
    return teacher

def build_student(coco_checkpoint_path,device):
    print("\nBuilding 62-class student...")
    checkpoint = torch.load(coco_checkpoint_path, map_location="cpu",weights_only=True)

    if checkpoint["num_classes"] != OLD_CLASSES:
        raise ValueError("COCO checkpoint must contain exactly 53 classes.")
    student = deeplabv3_resnet50(  weights=None, weights_backbone=None, aux_loss=False)
    in_channels = student.classifier[-1].in_channels

    student.classifier[-1] = nn.Conv2d( in_channels, OLD_CLASSES, kernel_size=1 )
    student.load_state_dict( checkpoint["model_state_dict"])
    old_head = student.classifier[-1]
    new_head = nn.Conv2d( in_channels, TOTAL_CLASSES, kernel_size=1)

    with torch.no_grad():
        new_head.weight[:OLD_CLASSES].copy_(  old_head.weight   )
        new_head.bias[:OLD_CLASSES].copy_(  old_head.bias  )

    student.classifier[-1] = new_head
    student.to(device)
    print(  f"Student classifier expanded: {OLD_CLASSES} -> {TOTAL_CLASSES}")
    return student

def knowledge_distillation_loss( student_logits, teacher_logits, valid_labels, temperature=2.0, confidence_threshold=0.60):
    """
    KD between:
        teacher: 53 classes
        student: first 53 of 62 classes
    KD is applied primarily where IDD has no direct semantic supervision.
    valid_labels:
        62-class IDD ground truth
        255 = ignore
    New IDD classes 53-61 are excluded from KD.
    Teacher confidence filtering avoids forcing the
    student to imitate low-confidence teacher predictions.
    """

    student_old = student_logits[:, :OLD_CLASSES]
    teacher_prob = F.softmax( teacher_logits / temperature, dim=1 )
    teacher_confidence, _ = teacher_prob.max( dim=1 )

    kd_mask = (        (valid_labels == IGNORE_INDEX) & (teacher_confidence >= confidence_threshold) )

    if kd_mask.sum() == 0:
        return student_logits.new_tensor(0.0)

    student_log_prob = F.log_softmax( student_old / temperature, dim=1 )

    kd_map = F.kl_div(        student_log_prob, teacher_prob, reduction="none").sum(dim=1)
    kd_loss = kd_map[kd_mask].mean()
    kd_loss = kd_loss * (temperature ** 2)
    return kd_loss

def compute_total_loss(student_logits,teacher_logits,labels,ce_criterion,kd_lambda=1.0,temperature=2.0,confidence_threshold=0.60):
    ce_loss = ce_criterion(student_logits,labels)
    kd_loss = knowledge_distillation_loss( student_logits=student_logits, teacher_logits=teacher_logits,    valid_labels=labels, temperature=temperature,  confidence_threshold=confidence_threshold)

    total_loss = ( ce_loss + kd_lambda * kd_loss)
    return total_loss, ce_loss, kd_loss

@torch.no_grad()
def evaluate( student,  val_loader, device):

    student.eval()
    confusion = torch.zeros( TOTAL_CLASSES, TOTAL_CLASSES, dtype=torch.int64 )
    total_loss = 0.0
    batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    for images, labels in val_loader:
        images = images.to( device, non_blocking=True  )

        labels = labels.to( device, non_blocking=True )

        with torch.amp.autocast( "cuda", enabled=(device.type == "cuda") ):
            outputs = student(images)
            logits = outputs["out"]
            loss = criterion( logits, labels )
        total_loss += loss.item()
        batches += 1
        predictions = logits.argmax( dim=1 ).cpu()
        labels_cpu = labels.cpu()
        valid = ( labels_cpu != IGNORE_INDEX)

        p = predictions[valid]
        l = labels_cpu[valid]

        if p.numel() > 0:
            indices = ( l * TOTAL_CLASSES + p )
            confusion += torch.bincount( indices,   minlength=TOTAL_CLASSES ** 2    ).reshape( TOTAL_CLASSES, TOTAL_CLASSES )

    intersection = confusion.diag().float()

    union = ( confusion.sum(dim=0).float() + confusion.sum(dim=1).float() - intersection )

    iou = intersection / union.clamp( min=1)
    valid_classes = union > 0
    miou = iou[valid_classes].mean().item()
    old_valid = (  valid_classes[:OLD_CLASSES] )
    new_valid = ( valid_classes[OLD_CLASSES:] )
    old_miou = ( iou[:OLD_CLASSES][old_valid].mean().item()
        if old_valid.any()
        else 0.0
    )

    new_miou = ( iou[OLD_CLASSES:][new_valid].mean().item()
        if new_valid.any()
        else 0.0
    )

    return ( total_loss / max(batches, 1), miou, old_miou, new_miou, iou )

def train_incremental( idd_root, coco_checkpoint_path, train_split="train", val_split="val", epochs=40,
    batch_size=4, image_size=512, lr_backbone=5e-6, lr_aspp=5e-5, lr_head=1e-3,
    kd_lambda=1.0, kd_temperature=2.0, kd_confidence=0.60, patience=7, min_delta=0.001, device="cuda",

    out_path="deeplabv3_coco_idd_62.pt",
    best_path="deeplabv3_coco_idd_62_best.pt",
    log_csv="deeplabv3_coco_idd_62_training.csv", img_ext=".jpg"):

    device = torch.device(device)
    print("\n" + "=" * 70)
    print("INCREMENTAL COCO -> IDD TRAINING")
    print("=" * 70)
    print(f"Old classes       : {OLD_CLASSES}")
    print(f"New classes       : {NEW_CLASSES}")
    print(f"Final classes     : {TOTAL_CLASSES}")
    print(f"KD lambda         : {kd_lambda}")
    print(f"KD temperature    : {kd_temperature}")
    print(f"KD confidence     : {kd_confidence}")
    print("=" * 70)

    train_images_dir = os.path.join( idd_root, train_split,  "images" )
    train_json_dir = os.path.join( idd_root, train_split, "json" )
    val_images_dir = os.path.join( idd_root, val_split, "images" )

    val_json_dir = os.path.join( idd_root, val_split, "json" )

    for path in [ train_images_dir,train_json_dir, val_images_dir, val_json_dir ]:

        if not os.path.isdir(path):
            raise FileNotFoundError( f"\nRequired directory not found:\n{path}\n\n"
                "Change train_split / val_split or idd_root." )

    scan_idd_labels( train_json_dir )

    train_stems = sorted([ f[:-5] for f in os.listdir(train_json_dir) if f.endswith(".json")])
    val_stems = sorted([f[:-5] for f in os.listdir(val_json_dir) if f.endswith(".json") ])

    print(f"\nIDD dataset:\n  Train: {len(train_stems)}\n  Val  : {len(val_stems)}")

    if len(train_stems) == 0:
        raise RuntimeError("No training JSON files found.")

    if len(val_stems) == 0:
        raise RuntimeError("No validation JSON files found.")

    train_dataset = IDDIncrementalDataset( images_dir=train_images_dir, json_dir=train_json_dir, file_stems=train_stems,
                                         image_size=image_size, img_ext=img_ext )

    val_dataset = IDDIncrementalDataset( images_dir=val_images_dir, json_dir=val_json_dir, file_stems=val_stems,
                                         image_size=image_size, img_ext=img_ext)

    train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True, persistent_workers=True)

    val_loader = DataLoader( val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        drop_last=False, persistent_workers=True )

    teacher = build_teacher( coco_checkpoint_path, device )

    student = build_student( coco_checkpoint_path, device )

    head_children = list( student.classifier.children() )

    aspp_parameters = []
    for layer in head_children[:-1]:
        aspp_parameters += list(layer.parameters())

    optimizer = torch.optim.AdamW([
            { "params": student.backbone.parameters(), "lr": lr_backbone},
            { "params": aspp_parameters, "lr": lr_aspp },
            { "params": student.classifier[-1].parameters(), "lr": lr_head },
        ], weight_decay=0.01)

    scheduler = ( torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=epochs ) )

    ce_criterion = nn.CrossEntropyLoss( ignore_index=IGNORE_INDEX )
    use_cuda = ( device.type == "cuda" )
    scaler = torch.amp.GradScaler( "cuda", enabled=use_cuda )

    log_file_exists = os.path.exists( log_csv )
    log_file = open( log_csv, "a", newline="" )
    writer = csv.writer( log_file)

    if not log_file_exists:

        writer.writerow([
            "epoch",
            "train_total_loss",
            "train_ce_loss",
            "train_kd_loss",
            "val_loss",
            "val_miou",
            "old_53_miou",
            "new_9_miou",
            "epoch_time_sec",
        ])

    best_miou = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        student.train()
        total_train_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0


        for step, (images,labels) in enumerate(train_loader):
            images = images.to( device, non_blocking=True)
            labels = labels.to( device, non_blocking=True)

            optimizer.zero_grad( set_to_none=True )

            with torch.amp.autocast( "cuda", enabled=use_cuda ):
                with torch.no_grad():
                    teacher_output = teacher(images)["out"]

                student_output = student( images)["out"]

                loss, ce_loss, kd_loss = (
                    compute_total_loss( student_logits=student_output, teacher_logits=teacher_output,
                        labels=labels, ce_criterion=ce_criterion,
                        kd_lambda=kd_lambda, temperature=kd_temperature, confidence_threshold=kd_confidence )
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kd_loss += kd_loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch:02d} Step {step:04d}/{len(train_loader)} Total={loss.item():.4f} CE={ce_loss.item():.4f} KD={kd_loss.item():.4f}")

        n_train = max(len(train_loader),1)
        avg_train_loss = (total_train_loss / n_train)
        avg_ce_loss = ( total_ce_loss / n_train )

        avg_kd_loss = ( total_kd_loss / n_train )

        ( val_loss, val_miou, old_miou, new_miou, iou_per_class ) = evaluate( student, val_loader, device )

        scheduler.step()
        epoch_time = ( time.time() - epoch_start )

        print("\n" + "-" * 70)
        print( f"Epoch {epoch}/{epochs}\nTrain total loss : {avg_train_loss:.4f}\nTrain CE loss    : {avg_ce_loss:.4f}\nTrain KD loss    : {avg_kd_loss:.4f}")

        print(f"Val loss         : {val_loss:.4f}\nVal mIoU         : {val_miou:.4f}\nOld 53 mIoU      : {old_miou:.4f}\nNew 9 mIoU        : {new_miou:.4f}\nEpoch time       : {epoch_time:.1f} sec")
        print("-" * 70)

        print("\nPer-class IoU:")
        for class_id in range(TOTAL_CLASSES):
            union_present = ( iou_per_class[class_id] > 0 )

            if union_present:
                print(f"{class_id:02d} {FINAL_CLASSES[class_id]:30s} {iou_per_class[class_id].item():.4f}")

        checkpoint = {
            "model_state_dict": student.state_dict(),
            "num_classes": TOTAL_CLASSES,
            "old_num_classes": OLD_CLASSES,
            "new_num_classes": NEW_CLASSES,
            "final_classes": FINAL_CLASSES,
            "idd_to_global": IDD_TO_GLOBAL,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "old_53_miou": old_miou,
            "new_9_miou": new_miou,
            "kd_lambda": kd_lambda,
            "kd_temperature": kd_temperature,
            "kd_confidence": kd_confidence,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        torch.save( checkpoint, out_path )
        print( f"\nCheckpoint saved:\n {out_path}")

        if val_miou > (best_miou + min_delta):
            best_miou = val_miou
            epochs_without_improvement = 0
            torch.save( checkpoint, best_path)
            print( f"BEST MODEL UPDATED\n mIoU = {best_miou:.4f}\n{best_path}")
        else:
            epochs_without_improvement += 1
            print(f"No meaningful improvement ({epochs_without_improvement}/{patience})")

        writer.writerow([
            epoch, round(avg_train_loss, 6), round(avg_ce_loss, 6), round(avg_kd_loss, 6), round(val_loss, 6),
            round(val_miou, 6), round(old_miou, 6), round(new_miou, 6), round(epoch_time, 2),
        ])
        log_file.flush()
        if epochs_without_improvement >= patience:
            print( "\nEarly stopping.")
            break
    log_file.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation mIoU : {best_miou:.4f}")
    print( f"Best model           : {best_path}")

if __name__ == "__main__":
    print_final_mapping()
    train_incremental(
        idd_root="./IDDPart2", coco_checkpoint_path= "deeplabv3_coco_stuff_best.pt", train_split="train",
        val_split="val", epochs=40, batch_size=4, image_size=512, lr_backbone=5e-6, lr_aspp=5e-5, lr_head=1e-3,
        kd_lambda=1.0, kd_temperature=2.0, kd_confidence=0.60, patience=7, min_delta=0.001, device="cuda",
        out_path= "deeplabv3_coco_idd_62.pt", best_path= "deeplabv3_coco_idd_62_best.pt",
        log_csv= "deeplabv3_coco_idd_62_training.csv", img_ext=".jpg",
    )