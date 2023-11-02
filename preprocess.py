import os
import glob
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import argparse
from pathlib import Path


def parse_csv(attribute_path):
    # attr_dict = {}
    attri_data = pd.read_csv(attribute_path)
    attri_data = attri_data[["name", "mos", "is_test"]]
    # attri_data_train = attri_data.query('is_test == False')
    # attr_lists_train = attri_data_train.to_dict('records')
    # attr_dict_train = {line['flickr_id']: line['mos'] for line in attr_lists_train}
    print(f"load data done.")
    return attri_data


def parse_excel_LBVD(attribute_path):

    attri_data = pd.read_excel(attribute_path, header=None)
    attri_data.columns = ["mos", "varience"]
    attri_data['name'] = range(1, len(attri_data)+1)

    print(f"load data done.")
    return attri_data


def parse_excel_Gaming(attribute_path):

    attri_data = pd.read_excel(attribute_path, header=None)
    attri_data.columns = ["name", "mos"]

    print(f"load data done.")
    return attri_data


def parse_excel_ugc(attribute_path):

    attri_data = pd.read_excel(attribute_path, sheet_name="MOS")
    attri_data = attri_data[["vid", "MOS full"]]
    attri_data.columns = ["name", "mos"]

    print(f"load data done.")
    return attri_data


def parse_prompts(prompt_path_prefix):
    iqa_prompts = open(
        os.path.join(
            prompt_path_prefix,
            'prompts/iqa_prompts.txt'
        ), "r").readlines()

    iqa_task_desc_prompts = open(
        os.path.join(
            prompt_path_prefix,
            'prompts/iqa_task_descriptor.txt'
        ), "r").readlines()

    return iqa_prompts, iqa_task_desc_prompts


def pd2dict(iqa_score, config):
    # print(iqa_score.loc[0:1])
    data_list = iqa_score.to_dict('records')
    print(data_list[0:1])
    if config.dataset == 'LBVD':
        data_dict = {str(line['name']): [line['task_descript'] +
                                         ' ' + line['questions'], line['mos']] for line in data_list}
    elif config.dataset == 'LIVE-YT-Gaming' or config.dataset == 'LIVE-VQC':
        data_dict = {eval(line['name']): [
            line['task_descript'] + ' ' + line['questions'], line['mos']] for line in data_list}
    else:
        data_dict = {line['name']: [line['task_descript'] + ' ' +
                                    line['questions'], line['mos']] for line in data_list}
    return data_dict


def save_js(data, path):
    with open(path, "wt") as f:
        json.dump(data, f)


def IQA(config):
    # iqa_img_files = os.listdir(iqa_img_path)
    # iqa_scores = os.path.join(iqa_scores_path, 'KoNViD_1k_mos.csv')
    # iqa_score = parse_csv(iqa_scores)
    if config.dataset == 'Konvid-1k':
        print('The current model is ' + config.dataset)
        iqa_score = pd.read_csv(config.metadata_path)  # "flickr_id","mos"
        print(f"load data done.")

    elif config.dataset == 'LSVQ':
        print('The current model is ' + config.dataset)
        iqa_score = pd.read_csv(config.metadata_path)
        iqa_score = iqa_score[["name", "mos", "is_test"]]
        print(f"load data done.")

    elif config.dataset == 'LBVD':
        print('The current model is ' + config.dataset)
        iqa_score = parse_excel_LBVD(config.metadata_path)
        print(f"load data done.")

    elif config.dataset == 'LIVE-VQC' or config.dataset == 'LIVE-YT-Gaming':
        print('The current model is ' + config.dataset)
        iqa_score = pd.read_excel(config.metadata_path, header=None)
        iqa_score.columns = ["name", "mos"]
        print(f"load data done.")

    elif config.dataset == 'YT-ugc':
        print('The current model is ' + config.dataset)
        iqa_score = parse_excel_ugc(config.metadata_path)
        print(f"load data done.")

    print(iqa_score.loc[0:1])
    now_path = os.path.dirname(os.path.abspath(__file__))
    iqa_prompts, iqa_task_desc_prompts = parse_prompts(now_path)
    out_js_name = now_path / Path('result') / Path(config.dataset)
    # iqa_prompts, iqa_task_desc_prompts = parse_prompts(os.getcwd())
    # out_js_name = os.getcwd() /  Path(config.dataset)

    iqa_score.insert(iqa_score.shape[1], 'questions', None)
    iqa_score.insert(iqa_score.shape[1], 'task_descript', None)

    # out_js = {}
    for i in tqdm(range(len(iqa_score))):
        iqa_score.loc[i, 'questions'] = np.random.choice(iqa_prompts).strip()
        iqa_score.loc[i, 'task_descript'] = np.random.choice(
            iqa_task_desc_prompts).strip()

    if 'is_test' in iqa_score:
        print('output training/testing set metadata')
        iqa_score_train = iqa_score.query('is_test == False')
        iqa_score_test = iqa_score.query('is_test == True')

    else:
        iqa_score_train = iqa_score.sample(n=int(len(iqa_score)*0.8))
        iqa_score_test = iqa_score.drop(iqa_score_train.index)
    
    if 'result' not in os.listdir(now_path):
        os.mkdir('result')

    data_dict_train = pd2dict(iqa_score_train, config)
    out_js_path = str(out_js_name) + "_train.json"
    save_js(data_dict_train, out_js_path)

    data_dict_test = pd2dict(iqa_score_test, config)
    out_js_path = str(out_js_name) + "_test.json"
    save_js(data_dict_test, out_js_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset', type=str,
                        help='Konvid-1k or LSVQ or LBVD or LIVE-VQC or LIVE-YT-Gaming or YT-ugc', default=None)
    parser.add_argument('--metadata_path', type=str, default=None)

    config = parser.parse_args()

    IQA(config)
