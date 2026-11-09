"""Relative depth of one picture with Depth Anything V2 (small), for matter.py.

    python matter_depth.py <image> <out.npy>

Runs under the anaconda Python (it has transformers and the cached weights); the
output is a float32 array the size of the image, larger = nearer (the model's
relative inverse depth), unnormalised. Offline: the weights must already be in
the Hugging Face cache (they are, from the room-walk depth route).
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    img = Image.open(src).convert("RGB")
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    with torch.no_grad():
        out = model(**proc(images=img, return_tensors="pt"))
        pred = out.predicted_depth
    d = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img.size[::-1], mode="bicubic",
                                        align_corners=False)[0, 0].numpy().astype(np.float32)
    np.save(dst, d)
    print(f"depth {d.shape[1]}x{d.shape[0]}: {d.min():.2f}..{d.max():.2f}")


if __name__ == "__main__":
    main()
