"""
@Time: 2023/10/11
@Author: cynthiazhang@sjtu.edu.cn
@Affiliation: SJTU
@Descriptor: load prompt, initial model configuration

"""
import os
import pickle
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from torchvision import transforms

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from pipeline.common.registry import registry
from torch_geometric.data import Data, Batch
from .conversation import Chat

from dataset.smiles2graph_demo import smiles2graph
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class ModelPipeline(Chat):

    def __init__(self, model, vis_processor=None, device='cuda:0'):
        super().__init__(model, vis_processor, device)

    def get_context_emb(self, conv, img_list=None):
        prompt = conv.get_prompt()  # for cat
        # print(f"prompts are: {prompt}")
        prompt_segs = prompt.split('<compoundHere>')
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        # mixed_embs = torch.cat(mixed_embs, dim=1)
        mix_embs = torch.sum(seg_embs[-1], dim=1)
        return seg_embs, mix_embs

    def answer(self, conv, img_list=None, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        embs, mix_embs = self.get_context_emb(conv, img_list)
        # current_max_len = embs.shape[1] + max_new_tokens
        # if current_max_len - max_length > 0:
        #     print('Warning: The number of tokens in current conversation exceeds the max length. '
        #           'The model will not see the contexts outside the range.')
        # begin_idx = max(0, current_max_len - max_length)
        #
        # embs = embs[:, begin_idx:]
        #
        # outputs = self.model.llama_model.generate(
        #     inputs_embeds=embs,
        #     max_new_tokens=max_new_tokens,
        #     stopping_criteria=self.stopping_criteria,
        #     num_beams=num_beams,
        #     do_sample=False,
        #     min_length=min_length,
        #     top_p=top_p,
        #     repetition_penalty=repetition_penalty,
        #     length_penalty=length_penalty,
        #     temperature=temperature,
        # )
        # output_token = outputs[0]
        # if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        #     output_token = output_token[1:]
        # if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        #     output_token = output_token[1:]
        # output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # output_text = output_text.split('###')[0]  # remove the stop sign '###'
        # output_text = output_text.split('Assistant:')[-1].strip()
        # conv.messages[-1][1] = output_text
        # return output_text, output_token.cpu().numpy()
        return mix_embs
