import os
from PIL import Image
import webdataset as wds
from pipeline.datasets.datasets.base_dataset import BaseDataset
from pipeline.datasets.datasets.caption_datasets import CaptionDataset
import pickle
from torch.utils.data import Dataset


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

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
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
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
            out = [xx for xx in out if xx["abstract"] and len(xx["abstract"]) > 0]
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

        t_graph = {
            'reac_idx': reac_idx, 'prod_idx': prod_idx,
            'rxn_idx': full_rxn_idx, 'reag_idx': rxn_nreg_idx,
            'batch_size': len(samples)
        }
        if len(reac_idx) > 0:
            t_graph['reactants'] = convert_graph_to_Data(reac_graphs)
        if len(prod_idx) > 0:
            t_graph['products'] = convert_graph_to_Data(prod_graphs)
        if len(full_rxn_idx) > 0:
            t_graph['rxn'] = {
                k: convert_graph_to_Data(v)
                for k, v in full_rxn.items()
            }
        if len(rxn_nreg_idx) > 0:
            t_graph['reagents'] = {
                k: convert_graph_to_Data(v)
                for k, v in rxn_wo_reg.items()
            }
        out = {'text_input': text_input, 'graph': t_graph}
        if len(questions) > 0:
            out['question'] = questions

        return out

    def __getitem__(self, index):
        if not self.qa_mode:
            return self.__getitem__0(index)
        else:
            return self.__getitem__1(index)

    def __getitem__0(self, index):
        # rec = self.data[index]
        # graph = rec["graph"]
        # caption = rec["abstract"]

        # return {
        #     "graph": graph,
        #     "text_input": caption,
        #     "_id": index,
        # }
        result = {
            k: v for k, v in self.data[index].items()
            if k != 'abstract'
        }
        result['_id'] = index
        result['text_input'] = self.data[index]['abstract']
        return result

    def __getitem__1(self, index):
        result = {
            k: v for k, v in self.data[index].items()
            if k != 'answer'
        }
        result['_id'] = index
        result['text_input'] = self.data[index]['answer']

        return result


def convert_graph_to_Data(data_batch):
    batch_size, max_node = len(data_batch), 0
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    batch, ptr, node_per_graph =  [], [0], []
    for idx, graph in enumerate(data_batch):
        num_nodes = graph['num_nodes']
        num_edges = graph['edge_index'].shape[1]

        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])

        lstnode += num_nodes
        max_node = max(max_node, num_nodes)
        node_per_graph.append(num_nodes)
        batch.append(np.ones(num_nodes, dtype=np.int64) * idx)
        ptr.append(lstnode)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0),
        'ptr': np.array(ptr, dtype=np.int64)
    }

    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    all_batch_mask = torch.zeros((batch_size, max_node))
    for idx, mk in enumerate(node_per_graph):
        all_batch_mask[idx, :mk] = 1
    result['batch_mask'] = all_batch_mask.bool()

    return torch_geometric.data.Data(**result)