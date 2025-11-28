import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

import argparse
import torch
from PIL import Image
import open_clip

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    path_to_model = "./sa_0_4_"+clip_model+"_linear.pth"

    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

amodel= get_aesthetic_model(clip_model="vit_l_14")
amodel.eval()

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                        help='dirs to input image')
    args = parser.parse_args()
    return args

def prepare_image_inputs(image_dir):
    image_list = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        image = preprocess(Image.open(img_path))
        image_list.append(image)
    image_inputs = torch.stack(image_list, dim=0)
    return image_inputs

# image = preprocess(Image.open("lovely-cat-as-domestic-animal-view-pictures-182393057.jpg")).unsqueeze(0)
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     prediction = amodel(image_features)
#     print(prediction)

if __name__ == "__main__":
    args = parse_arg()
    image_inputs = prepare_image_inputs(args.image_dir)
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        predictions = amodel(image_features)
        print("Aesthetic Score:", predictions.mean())