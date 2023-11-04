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
import json

from pipeline.common.registry import registry
from torch_geometric.data import Data, Batch

from dataset.smiles2graph_demo import smiles2graph
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def parse_json(path):
    with open(path) as r:
        data = r.readlines()
    data_length = len(data)
    meta_data = []
    for i in range(data_length):
        meta_data.append(json.loads(data[i]))
    return meta_data


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


class Chat:
    def __init__(self, model, vis_processor=None, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</compound>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
            print(conv.messages)
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        print(f"conv.message is {conv.messages}")
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=False,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image_folder, conv, img_list, autocast=False, autocast_proj=False):
        # assert isinstance(image, str), f"Expected a string but got {image}"
        inputs = {}
        for img_path in os.listdir(image_folder):
            img_save_path = os.path.join(image_folder, img_path)
            img = Image.open(img_save_path).convert("RGB")
        inputs["image"] = self.transforms(img).unsqueeze(0).to(self.device)

        image_emb, _ = self.model.encode_img_infer(inputs, device=self.device, autocast=autocast,
                                                   autocast_proj=autocast_proj)
        print(image_emb.shape)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<compound><compoundHere></compound>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def upload_graph_single(self, smiles, conv, img_list, autocast=False, autocast_proj=False):
        assert isinstance(smiles, str), f"Expected a string but got {smiles}"
        timestamp = time.time()
        inputs = {}
        with open("dataset/tmp_smiles.txt", "wt") as f:
            f.write(str(timestamp) + " " + smiles)
        g = smiles2graph(smiles)
        graph0 = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']),
                      edge_attr=torch.asarray(g['edge_feat']))
        inputs["graph"] = Batch.from_data_list([graph0]).to(self.device)

        # for _ in range(60):
        #     time.sleep(1)
        #     pkl_file = "dataset/tmp_smiles.pkl"
        #     if os.path.isfile(pkl_file):
        #         with open(pkl_file, "rb") as f:
        #             res = pickle.load(f)
        #         t2 = res["timestamp"]
        #         if t2 > timestamp:
        #             if "graph" in res:
        #                 g = res["graph"]
        #                 graph0 = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']),
        #                               edge_attr=torch.asarray(g['edge_feat']))
        #                 inputs["graph"] = Batch.from_data_list([graph0]).to(self.device)
        #             if "img_save_path" in res:
        #                 img_save_path = res["img_save_path"]
        #                 img = Image.open(img_save_path).convert("RGB")
        #                 inputs["image"] = self.transforms(img).unsqueeze(0).to(self.device)
        #             break

        image_emb, _ = self.model.encode_img_infer(inputs, device=self.device, autocast=autocast,
                                                   autocast_proj=autocast_proj)
        img_list.append(image_emb)  # image_emb.shape: [1, 1, 4096]
        conv.append_message(conv.roles[0], "<compound><compoundHere></compound>")
        msg = "Received."
        return msg

    def get_context_emb(self, conv, img_list=None):
        prompt = conv.get_prompt()
        print(f"prompts is: {prompt}")
        prompt_segs = prompt.split('<compoundHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)

        return mixed_embs  # [1, 173, 4096]
