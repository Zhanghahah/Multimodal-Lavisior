import argparse
import json
import random
import copy
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pipeline.common.config import Config
from pipeline.common.dist_utils import get_rank
from pipeline.common.registry import registry
from pipeline.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from pipeline.datasets.builders import *
from pipeline.models import *
from pipeline.processors import *
from pipeline.runners import *
from pipeline.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1, help="specify the num_beams for text generation.")
    parser.add_argument("--temperature", type=float, default=1, help="specify the temperature for text generation.")
    parser.add_argument("--out_file", type=str, default="data/ChEMBL_QA_test_imgMol2.json", help="specify the output file.")
    parser.add_argument("--in_file", type=str, default="data/ChEMBL_QA_test.json", help="specify the output file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
use_amp = cfg.run_cfg.get("amp", False)
amp_encoder = cfg.run_cfg.get("amp_encoder", use_amp)
amp_proj = cfg.run_cfg.get("amp_proj", use_amp)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
print(model_config)
model = model_cls.from_config(model_config)

model = model.to('cuda:{}'.format(args.gpu_id))

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

@torch.no_grad()
def upload_img(gr_img):
    assert gr_img is not None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_graph_single(gr_img, chat_state, img_list, autocast=amp_encoder, autocast_proj=amp_proj)
    return chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    assert len(user_message) != 0
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


@torch.no_grad()
def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    with torch.cuda.amp.autocast(use_amp):
        llm_message = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list

def parse_json(path):
    with open(path) as r:
        data = r.readlines()
    data_length = len(data)
    meta_data = []
    for i in range(data_length):
        meta_data.append(json.loads(data[i]))
    return meta_data

def infer(smiles, question):
    
    chat = []
    chat_state, img_list = upload_img(smiles)
    

    chatbot = []
    chat_state_ = copy.deepcopy(chat_state)
    text_input, chatbot, chat_state_ = gradio_ask(question, chatbot, chat_state_)
    chatbot, chat_state_, img_list_ = gradio_answer(chatbot, chat_state_, img_list, args.num_beams, args.temperature)
    chat.extend(chatbot)

    return chat


def infer_chembl_QA():
    test_samples = parse_json(args.in_file)
    out = {}
    total_length = len(test_samples)
    for i in tqdm(range(0, total_length)):
        data = test_samples[i]
        if data['type'] == 'products':
            smiles = data['input']  # smiles for reactants, products, or reagents
            smiles_ = copy.copy(smiles)
            question = data['instruction']
            answer = data['output']
            qa_pairs = infer(smiles, question)
            # assert len(qa_pairs) == len(answer)
            out[smiles_] = qa_pairs[0][-1]
        else:
            continue
    # for smi, rec in tqdm.tqdm(js.items()):
    #     t0 = time.time()
    #
    #     smi_ = copy.copy(smi)
    #     questions = [question for question, answer in rec]
    #     answers = [answer for question, answer in rec]
    #     qa_pairs = infer(smi, questions)
    #     assert len(qa_pairs) == len(answers)
    #     for ans, qa in zip(answers, qa_pairs):
    #         qa.insert(1, ans)
    #     out[smi_] = qa_pairs
    #     # print(smi_, "============ used time:", time.time() - t0)
    #
    #     for qa in qa_pairs:
    #         print(qa)

    with open(args.out_file, "wt") as f:
        json.dump(out, f)


infer_chembl_QA()