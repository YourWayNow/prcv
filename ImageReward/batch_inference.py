import os
import torch
import ImageReward as RM
import json
from PIL import Image

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="ImageReward Inference Example")
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--prompt_dict", type=str, help="Text prompt for scoring")
    return parser.parse_args()


if __name__ == "__main__":
    # img_prefix = "assets/images"
    # generations = [f"{pic_id}.webp" for pic_id in range(1, 5)]
    # img_list = [os.path.join(img_prefix, img) for img in generations]
    # prompt_list = ["a painting of an ocean with clouds and birds, day time, low depth field effect"] * len(generations)
    args = parse_args()
    img_dir = args.image_dir
    prompt_dict = json.load(open(args.prompt_dict, 'r'))

    img_list = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
    img_list_no_ext = [img_name.split(".")[0] for img_name in os.listdir(img_dir)]
    prompt_list = [prompt_dict[img_name] for img_name in img_list_no_ext]

    # import random
    # random.shuffle(img_list)

    model = RM.load("ImageReward-v1.0")
    with torch.no_grad():
        # ranking, rewards = model.inference_rank(prompt_list[0], img_list)
        ranking, rewards = model.inference_rank_multiple_prompts(prompt_list, img_list)
        # Print the result
        print("\nPreference predictions:\n")
        print(f"ranking = {ranking}")
        print(f"rewards = {rewards}")
        for index in range(len(img_list)):
            score = model.score(prompt_list[index], img_list[index])
            print(f"{img_list[index]:>16s}: {score:.2f}")
