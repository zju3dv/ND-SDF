import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from lang_sam import LangSAM
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import pil_to_tensor
import glob, os
from tqdm import tqdm
import argparse
def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    vis = os.path.join(output_dir, "vis")
    res = os.path.join(output_dir, "res")
    os.makedirs(vis, exist_ok=True)
    os.makedirs(res, exist_ok=True)

    img_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)
    img_paths.sort()
    text_prompt = args.prompt
    model = LangSAM()

    for image in tqdm(img_paths):
        
        vis_out = os.path.join(vis, os.path.basename(image))
        res_out = os.path.join(res, os.path.basename(image)[:-4]+'.npy')

        image_pil = Image.open(image).convert("RGB")
        img = pil_to_tensor(image_pil)
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
        mask = torch.zeros_like(img[0]).bool()
        # print(mask.shape)
        # quit()

        for seg in masks:
            if mask is not None:
                mask = torch.logical_or(mask, seg)
            else:
                mask = seg
        img = draw_segmentation_masks(img, mask, colors='cyan', alpha=.4)
        img = img.numpy().transpose(1, 2, 0)

        np.save(res_out, mask.numpy())
        plt.imsave(vis_out, img)
        # quit()
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i","--input_dir", required=True, type=str)
    args.add_argument("-o","--output_dir", default='./output', type=str)
    args.add_argument("--prompt", default="ceiling.wall.floor", type=str)
    args = args.parse_args()
    print(args)
    main(args)

