import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from src.hf_login import get_hf_token

class FGVCAircraftDataset(Dataset):
    ALLOWED = {"manufacturer", "family", "variant"}
    def __init__(
        self,
        root: str ,
        split: str = "train",
        level: str = "manufacturer",
        transform = None,
        return_class: bool = False,
        cropped: bool = False,
        album: bool = False,
    ):
        """
        root can be:
          - a local path: "fgvc-aircraft-2013b/data"
          - an HF Hub repo_id: "chocp/fgvc-aircraft-2013b"
        """
        assert level in self.ALLOWED, f"level must be one of {self.ALLOWED}"
        # 1) if root looks like an HF repo_id, pull it down
        if not os.path.isdir(root) and "/" in root:
            # clone the entire repo and return path to the `data/` subfolder
            repo_path = snapshot_download(root, repo_type="dataset", token=get_hf_token(), max_workers=1)
            root = os.path.join(repo_path, "data")
        self.root         = root
        self.split        = split
        self.level        = level
        self.transform    = transform
        self.return_class = return_class
        self.cropped      = cropped
        self.album        = album
        # 2) load image IDs + labels
        class_file = os.path.join(root, f"images_{level}_{split}.txt")
        with open(class_file) as f:
            lines = [ln.strip().split() for ln in f]
        self.samples = [(img_id, " ".join(lbls)) for img_id, *lbls in lines]
        # 3) classes :left_right_arrow: idx
        self.classes      = sorted({lbl for _, lbl in self.samples})
        self.class_to_idx = {lbl: i for i, lbl in enumerate(self.classes)}
        self.idx_to_class = {i: lbl for lbl, i in self.class_to_idx.items()}
        # 4) bounding boxes
        bbox_file = os.path.join(root, "images_box.txt")
        with open(bbox_file) as f:
            self.bboxes = {
                ln.split()[0]: tuple(map(int, ln.split()[1:]))
                for ln in f
            }
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_id, class_str = self.samples[idx]
        img_path = os.path.join(self.root, "images", f"{img_id}.jpg")
        img      = Image.open(img_path).convert("RGB")
        if self.cropped:
            xmin, ymin, xmax, ymax = self.bboxes[img_id]
            img = img.crop((xmin, ymin, xmax, ymax))
        if self.transform:
            if not self.album:
                img = self.transform(img)
            else:
                # albumentations expects numpy array
                img_np = np.array(img)
                img = self.transform(image=img_np)["image"]
        if self.return_class:
            return img, class_str
        else:
            return img, self.class_to_idx[class_str]



ds_hf = FGVCAircraftDataset(
    root="chocp/fgvc-aircraft-2013b",
    split="test", level="manufacturer", transform=None
)