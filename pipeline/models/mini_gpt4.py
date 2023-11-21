import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import clip

from pipeline.common.registry import registry
from pipeline.models.blip2 import Blip2Base, disabled_train
from pipeline.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from pipeline.models.gnn import GNN
import contextlib
from pipeline.models.base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList

from pipeline.models.image_mol import ImageMol
from pipeline.models.MultiStepGraph import AttentionPooling, RXNGAT


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def collate_feat_batch(x, batch_mask):
    batch_size, max_node = batch_mask.shape
    overall_feat = torch.zeros(batch_size, max_node, x.shape[-1])
    overall_feat = overall_feat.to(x)
    overall_feat[batch_mask] = x
    return overall_feat


@registry.register_model("mini_gpt4")
class MiniGPT4(BaseModel):
    """
    GNN GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/drugchat.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        encoder_ckpt=None,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,
        # use 8 bit and put vit in cpu
        # the device of 8bit model should be set
        # when loading and cannot be changed anymore.
        device_8bit=0,
        use_graph_agg=True,
        encoder_names=["gnn"],
        feat_dims=None,
        prompt_tuning=0,
        graph_dim=512,
        rxn_gat_layer=2,
        dropout=0.1,
        negative_slope=0.2,
        graph_backbone_dim=300,
        nheads=1,
        share_pool_weight=True,
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.encoder_names = encoder_names

        self.feat_dims = feat_dims
        if "gnn" in self.encoder_names:
            self.use_graph_agg = use_graph_agg
            self.share_pool_weight = share_pool_weight
            self.create_gnn(encoder_ckpt, freeze_qformer)
            self.rxn_gat = RXNGAT(
                n_layer=rxn_gat_layer, gnn_dim=graph_backbone_dim,
                heads=nheads, dropout=dropout, emb_dim=graph_dim,
                component_keys={'products', 'reactants', 'reagents'}
            )
            if self.use_graph_agg and not self.share_pool_weight:
                self.pooler = torch.nn.ModuleDict({
                    'reactants': AttentionPooling(
                        output_dim=graph_dim, num_heads=nheads,
                        dropout=dropout, emb_dim=graph_backbone_dim
                    ),
                    'products': AttentionPooling(
                        output_dim=graph_dim, num_heads=nheads,
                        dropout=dropout, emb_dim=graph_backbone_dim
                    )
                })
            if not self.use_graph_agg and graph_backbone_dim != graph_dim:
                self.g_align_proj = torch.nn.Linear(
                    graph_backbone_dim, graph_dim
                )
            else:
                self.g_align_proj = None

        if "image_mol" in self.encoder_names:
            self.create_image_mol(encoder_ckpt, freeze_qformer)

        self.ln_vision = nn.Identity()

        print('Loading LLAMA')
        # import pudb
        # pudb.set_trace()
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        if self.feat_dims is not None:
            self.llama_proj = nn.ModuleDict()
            for kk, dim in self.feat_dims.items():
                self.llama_proj.add_module(
                    kk,
                    nn.Linear(dim, self.llama_model.config.hidden_size)
                )
        else:
            self.llama_proj = nn.Linear(
                self.encoder_out_dim, self.llama_model.config.hidden_size
            )

        if prompt_tuning:
            self.register_parameter("soft_prompt", nn.Parameter(
                torch.randn(1, prompt_tuning, self.llama_model.config.hidden_size)))
            nn.init.uniform_(self.soft_prompt, -0.5, 0.5)
        else:
            self.soft_prompt = None
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts if "<compoundHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(
                p) for p in filted_prompts]
            # self.prompt_list = ["###Human: <compound><compoundHere></compound>###Human: Can you describe the mechanism of this drug?###Assistant:"]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(
                random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def create_gnn(self, model_path, freeze):
        print('Loading GNN')
        print(f"use_graph_agg={self.use_graph_agg}")
        self.gnn = GNN(num_layer=5, emb_dim=300, gnn_type='gcn')
        self.gnn.load_from_pretrained(url_or_filename=model_path)
        self.encoder_out_dim = self.gnn.out_dim

        if freeze:
            for name, param in self.gnn.named_parameters():
                param.requires_grad = False
            self.gnn = self.gnn.eval()
            self.gnn.train = disabled_train
            logging.info("freezed GNN")

        print('Loaded GNN')

    def create_image_mol(self, model_path, freeze):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, self.preprocess = clip.load("ViT-B/32", device=device)

        # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)

        # with torch.no_grad():
        #     image_features = model.encode_image(image)
        # model = ImageMol()
        # model.load_from_pretrained(url_or_filename=model_path)
        self.image_mol = model
        self.encoder_out_dim = model.visual.output_dim

        if freeze:
            for name, param in self.image_mol.named_parameters():
                param.requires_grad = False
            self.image_mol = self.image_mol.eval()
            self.image_mol.train = disabled_train
            logging.info("freezed image_mol")
        print('Loaded image_mol')

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        if "gnn" in self.encoder_names:
            self.gnn.to("cpu")
            self.gnn.float()
        if "image_mol" in self.encoder_names:
            self.image_mol.to("cpu")
            self.image_mol.float()

    def encode_molecules(self, inputs, device):
        key2result = {}
        for k, v in inputs.items():
            if k in ['reactants', 'products']:
                node_feat = self.gnn(v['graph'])
                if self.use_graph_agg:
                    if self.share_pool_weight:
                        batched_result = self.rxn_gat.Attn_pools[k](
                            graph_emb=node_feat, batch=v['graph'].batch
                        ).unsqueeze(dim=1)
                    else:
                        batched_result = self.pooler[k](
                            graph_emb=node_feat, batch=v['graph'].batch
                        ).unsqueeze(dim=1)
                    batch_mask = torch.ones(batched_result.shape[0], 1)
                    batch_mask = batch_mask.bool().to(device)
                else:
                    batched_result = collate_feat_batch(
                        x=node_feat, batch_mask=v['graph'].batch_mask
                    ).unsqueeze(dim=1)
                    batch_mask = v['graph'].batch_mask
                    if self.g_align_proj is not None:
                        batched_result = self.g_align_proj(batched_result)

            else:
                gat_kwargs = {
                    a: {'graph': b, 'embeddings': self.gnn(b)}
                    for a, b in v.items()
                }
                rxn_embs = self.rxn_gat(gat_kwargs)
                batched_result = rxn_embs['rxn'].unsqueeze(dim=1)
                batch_mask = torch.ones(batched_result.shape[0], 1)
                batch_mask = batch_mask.bool().to(device)
            key2result[k] = {'feat': batched_result, 'batch_mask': batch_mask}

        max_node = max(v['feat'].shape[1] for v in key2result.values())
        batch_size = inputs['batch_size']
        batch_mask = torch.zeros(batch_size, max_node).bool().to(device)
        overall_feat = torch.zeros(batch_size, max_node, self.graph_dim)
        overall_feat = overall_feat.to(device)

        if len(inputs['reac_idx']) > 0:
            this_mask = key2result['reactants']['batch_mask']
            this_feat = key2result['reactants']['feat']
            this_node = this_mask.shape[1]
            overall_feat[inputs['reac_idx'], :this_node] = this_feat
            batch_mask[inputs['reac_idx'], :this_node] = this_mask

        if len(inputs['prod_idx']) > 0:
            this_mask = key2result['products']['batch_mask']
            this_feat = key2result['products']['feat']
            this_node = this_mask.shape[1]
            overall_feat[inputs['prod_idx'], :this_node] = this_feat
            batch_mask[inputs['prod_idx'], :this_node] = this_mask

        if len(inputs['rxn_idx']) > 0:
            this_mask = key2result['rxn']['batch_mask']
            this_feat = key2result['rxn']['feat']
            this_node = this_mask.shape[1]
            overall_feat[inputs['rxn_idx'], :this_node] = this_feat
            batch_mask[inputs['rxn_idx'], :this_node] = this_mask

        if len(inputs['reag_idx']) > 0:
            this_mask = key2result['reagents']['batch_mask']
            this_feat = key2result['reagents']['feat']
            this_node = this_mask.shape[1]
            overall_feat[inputs['reag_idx'], :this_node] = this_feat
            batch_mask[inputs['reag_idx'], :this_node] = this_mask

        return overall_feat, batch_mask

    def encode_img(self, inputs, device, do_proj=True):
        """
        Args:
            inputs (dict)
        """
        if "gnn" in self.encoder_names:
            graph = inputs['graph']
            device = graph.x.device
            if self.low_resource:
                self.vit_to_cpu()
                graph = graph.to("cpu")

            graph_feat, batch_mask = self.encode_molecules(graph).to(device)  # ([1, 11, 512])
            feat = graph_feat
            inputs["feat"] = feat
            inputs["graph_feat"] = feat

        if do_proj:
            inputs_llama, atts_llama = self.proj_feat(inputs, device)
            return inputs_llama, atts_llama
        return inputs

    def encode_img_infer(self, inputs, device, autocast=False, autocast_proj=False):
        """
        Need this function to fix the inference data casting issues
        """
        with torch.cuda.amp.autocast(autocast):
            features = self.encode_img(inputs, device, do_proj=False)
        with torch.cuda.amp.autocast(autocast_proj):
            out = self.proj_feat(features, device)
        return out

    def proj_feat(self, features, device):
        """
        Args:
            features (dict)
        """
        if self.feat_dims is not None:  # none
            feats = []
            for kk, dim in self.feat_dims.items():
                feat = features[kk]
                inputs_tokens = self.llama_proj[kk](feat)
                feats.append(inputs_tokens)
            img_embeds = torch.cat(feats, dim=1)
            atts_img = torch.ones(
                img_embeds.size()[:-1], dtype=torch.long).to(device)
            return img_embeds, atts_img

        embeds = self.ln_vision(features["feat"]).to(
            device)  # torch.Size([1, 11, 300])

        inputs_llama = self.llama_proj(embeds)  # torch.Size([1, 11, 4096])
        # torch.Size([1, 11])
        atts_llama = torch.ones(
            inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama

    def pad_node(self, data, node_representation):
        # pad the repr so that each graph has some number of node repr
        ptr = data.ptr.tolist()
        nodes = [node_representation[ptr[i]: ptr[i+1]]
                 for i in range(data.num_graphs)]
        nnodes = [ptr[i+1] - ptr[i] for i in range(data.num_graphs)]
        max_len = max(nnodes)
        pad_size = [max_len - x_ for x_ in nnodes]
        pad = self.pad_token.to(device=node_representation.device)
        node_repr = torch.stack([torch.cat([node, pad.expand(pz, -1)])
                                 for pz, node in zip(pad_size, nodes)])
        return node_repr

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            batch_size = img_embeds.shape[0]
            ps = [prompt.split('<compoundHere>') for prompt in prompts]
            p_before, p_after = list(zip(*ps))
            p_before_tokens = self.llama_tokenizer(
                p_before, padding="longest", return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, padding="longest", return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            # .expand(batch_size, -1, -1) torch.Size([1, 8, 4096])
            p_before_embeds = self.llama_model.model.embed_tokens(
                p_before_tokens.input_ids)
            # .expand(batch_size, -1, -1)   torch.Size([1, 15, 4096])
            p_after_embeds = self.llama_model.model.embed_tokens(
                p_after_tokens.input_ids)
            if self.soft_prompt is not None:
                img_embeds = torch.cat(
                    [img_embeds, self.soft_prompt.expand(batch_size, -1, -1)], 1)
            wrapped_img_embeds = torch.cat(
                [p_before_embeds, img_embeds, p_after_embeds], dim=1)  # torch.Size([1, 34, 4096])
            # wrapped_atts_img 1,34
            wrapped_atts_img = atts_img[:,
                                        :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def get_device_from_graph(self, graph):
        for idx, key in enumerate(['reactants', 'products', 'rxn', 'reagents']):
            if key in graph and idx < 2:
                return graph[key].x.device
            elif key in graph:
                return graph[key]['reactants'].x.device
        raise ValueError('A batch any graph is passed')

    def forward(self, samples):
        if "gnn" in self.encoder_names:
            inputs = samples['graph']
            device = self.get_device_from_graph(inputs)
        if "image_mol" in self.encoder_names:
            inputs = samples["image"]
            device = inputs.device
        if "feat" in self.encoder_names:
            # no encoder
            device = list(v for v in samples.values()
                          if isinstance(v, torch.Tensor))[0].device

        img_embeds, atts_img = self.encode_img(inputs, device)

        assert 'question' in samples
        if 'question' in samples:
            # assert len(samples['question']) == 1, "not supporting batch mode yet"
            vqa_prompt = ['###Human: <compound><compoundHere></compound> ' +
                          qq + "###Assistant: " for qq in samples['question']]
            img_embeds, atts_img = self.prompt_wrap(
                img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(
                img_embeds, atts_img, [prompt])

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]  # noqa: answer

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(
            to_regress_tokens.input_ids)
        img_embeds = img_embeds.to(dtype=bos_embeds.dtype)
        inputs_embeds = torch.cat(
            [bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
        # embs = torch.cat([bos_embeds, img_embeds], dim=1)
        # tt = self.gen_(embs)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    def gen_(self, embs):
        """
        Generate text.
        """
        stop_words_ids = [torch.tensor([835]).to(embs.device),
                          torch.tensor([2277, 29937]).to(embs.device)]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=300,
            stopping_criteria=stopping_criteria,
            num_beams=1,
            do_sample=False,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.,
            length_penalty=1.,
            temperature=1,
        )
        output_token = outputs[0]
        # the model might output a unknow token <unk> at the beginning. remove it
        if output_token[0] == 0:
            output_token = output_token[1:]
        # some users find that there is a start token <s> at the beginning. remove it
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(
            output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        return output_text

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        encoder_ckpt = cfg.get("encoder_ckpt", "ckpt/gcn_contextpred.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        use_graph_agg = cfg.get("use_graph_agg", True)
        encoder_name = cfg.get("encoder_name", "gnn")
        encoder_names = cfg.get("encoder_names", [encoder_name])
        # a dict that controls the name of llama_proj
        feat_dims = cfg.get("feat_dims", None)
        prompt_tuning = cfg.get("prompt_tuning", 0)

        graph_dim = cfg.get('graph_dim', 512)
        rxn_gat_layer = cfg.get('rxn_gat_layer', 2)
        dropout = cfg.get('dropout', 0.1)
        negative_slope = cfg.get('negative_slope', 0.2)
        graph_backbone_dim = cfg.get('graph_backbone_dim', 300)
        nheads = cfg.get('nheads', 8)
        share_pool_weight = cfg.get('share_pool_weight', True)

        model = cls(
            vit_model=vit_model,
            encoder_ckpt=encoder_ckpt,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            use_graph_agg=use_graph_agg,
            encoder_names=encoder_names,
            feat_dims=feat_dims,
            prompt_tuning=prompt_tuning,
            graph_dim=graph_dim,
            rxn_gat_layer=rxn_gat_layer,
            dropout=dropout,
            negative_slope=negative_slope,
            graph_backbone_dim=graph_backbone_dim,
            nheads=nheads,
            share_pool_weight=share_pool_weight
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print("Loaded checkpoint from {}: {}".format(ckpt_path, msg))

        return model
