import os
import torch
import ImageReward as RM

if __name__ == "__main__":
    img_prefix = "assets/images"
    generations = [f"{pic_id}.webp" for pic_id in range(1, 5)]
    img_list = [os.path.join(img_prefix, img) for img in generations]
    prompt_list = ["a painting of an ocean with clouds and birds, day time, low depth field effect"] * len(generations)

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
            print(f"{generations[index]:>16s}: {score:.2f}")
