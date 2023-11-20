import os
from PIL import Image
import webdataset as wds
from pipeline.datasets.datasets.base_dataset import BaseDataset
from pipeline.datasets.datasets.caption_datasets import CaptionDataset
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super(CCSBUDataset, self).__init__(
            vis_processor=vis_processor,
            text_processor=text_processor
        )

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }


class CCSBUAlignDataset0(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class CCSBUAlignDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None,
        vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of data
        """
        self.vis_root = vis_root

        print(f"Open data file {vis_root}")
        with open(vis_root, "rb") as f:
            out = pickle.load(f)

        if "abstract" in out[0]:
            self.qa_mode = False
            out = [xx for xx in out if xx["abstract"]
                   and len(xx["abstract"]) > 0]
        elif "answer" in out[0]:
            self.qa_mode = True
            out = [xx for xx in out if xx["answer"] and len(xx["answer"]) > 0]

        self.data = out

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collater(samples):
        reac_graphs, reac_idx = [], []
        prod_graphs, prod_idx = [], []
        full_rxn = {"reactants": [], 'products': [], 'reagents': []}
        rxn_wo_reg = {"reactants": [], 'products': []}
        full_rxn_idx, rxn_nreg_idx = [], []
        text_input, questions = [], []

        for idx, x in enumerate(samples):
            if x['type'] == 'reactants':
                reac_graphs.append(x['graph'])
                reac_idx.append(idx)
            elif x['type'] == 'products':
                prod_graphs.append(x['graph'])
                prod_idx.append(idx)
            elif x['type'] in ['classification', 'yield']:
                full_rxn['reactants'].append(x['reactants'])
                full_rxn['products'].append(x['products'])
                full_rxn['reagents'].append(x['reagents'])
                full_rxn_idx.append(idx)
            elif x['type'] == 'reagents':
                rxn_wo_reg['reactants'].append(x['reactants'])
                rxn_wo_reg['products'].append(x['products'])
                rxn_nreg_idx.append(idx)
            else:
                raise NotImplementedError(f'Invalid type {x["type"]}')
            text_input.append(x['text_input'])
            if "question" in x:
                questions.append(x['question'])

        out = {
            'text_input': text_input, 'graph': {
                'reactants': Batch.from_data_list(reac_graphs),
                'reac_idx': reac_idx,
                'products': Batch.from_data_list(prod_graphs),
                'prod_idx': prod_idx,
                'rxn': {
                    k: Batch.from_data_list(v)
                    for k, v in full_rxn.items()
                },
                'rxn_idx': full_rxn_idx,
                'reagents': {
                    k: Batch.from_data_list(v)
                    for k, v in rxn_wo_reg.items()
                },
                'reag_idx': rxn_nreg_idx,
                'batch_size': len(samples)
            }
        }
        if len(questions) > 0:
            out['question'] = questions

        return out

    def __getitem__(self, index):
        if not self.qa_mode:
            return self.__getitem__0(index)
        else:
            return self.__getitem__1(index)

    def __getitem__0(self, index):
        rec = self.data[index]
        graph = rec["graph"]
        caption = rec["abstract"]

        return {
            "graph": graph,
            "text_input": caption,
            "_id": index,
        }

    def __getitem__1(self, index):
        import pudb
        pudb.set_trace()
        result = {
            k: v for k, v in self.data[index].items()
            if k != 'answer'
        }
        result['_id'] = index
        result['text_input'] = self.data[index]['answer']

        return result
