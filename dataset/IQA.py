import os
import glob
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def parse_csv(attribute_path):
    attr_dict = {}
    attri_data = pd.read_csv(attribute_path)
    attr_lists = attri_data.to_dict('records')
    attr_dict = {line['flickr_id']: line['mos'] for line in attr_lists}
    print(f"load data done.")
    return attr_dict


def parse_prompts(prompt_path_prefix):
    iqa_prompts = open(
        os.path.join(
            prompt_path_prefix,
            'iqa_prompts.txt'
        ), "r").readlines()

    iqa_task_desc_prompts = open(
        os.path.join(
            prompt_path_prefix,
            'iqa_task_descriptor.txt'
        ), "r").readlines()

    return iqa_prompts, iqa_task_desc_prompts


def IQA(iqa_img_path, iqa_scores_path):
    iqa_img_files = os.listdir(iqa_img_path)
    iqa_scores = os.path.join(iqa_scores_path, 'KoNViD_1k_mos.csv')
    iqa_score_dict = parse_csv(iqa_scores)

    iqa_prompts, iqa_task_desc_prompts = parse_prompts(os.getcwd())

    out_js_path = os.path.join(os.getcwd(), "img_qa_test.json")

    out_js = {}
    for img_folder in tqdm(iqa_img_files[:100]):
        questions = np.random.choice(iqa_prompts).strip()
        task_descript = np.random.choice(iqa_task_desc_prompts).strip()
        full_prompts = task_descript + ' ' + questions
        answer = iqa_score_dict[int(img_folder)]
        out_js[img_folder] = [full_prompts, answer]

    with open(out_js_path, "wt") as f:
        json.dump(out_js, f)


if __name__ == '__main__':
    dataset_path_prefix = '/home/zhangyu/data/1K/'
    iqa_img_path = os.path.join(dataset_path_prefix, 'KoNViD_1k_images')
    iqa_scores_path = os.path.join(dataset_path_prefix, 'metadata/')
    IQA(iqa_img_path, iqa_scores_path)
