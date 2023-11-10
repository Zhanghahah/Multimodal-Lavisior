import pickle
import torch
import os
from tqdm import tqdm
from torch_geometric.data import Data, Batch



def main(list_of_path, path_prefix):
    out = []

    for path in list_of_path:
        path = os.path.join(path_prefix, path)
        with open(path, "rb") as f:
            ret = pickle.load(f)
        if "question" in ret[0]:
            # {"graph": graph, "question": question, "answer": str(answer)}
            total_length = len(ret)
            for i in tqdm(range(0, total_length)):
                rec = ret[i]
                g = rec["graph"]
                graph = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
                rec["graph"] = graph
                out.append(rec)

        else:
            # {"graph": graph, "abstract": abstract}
            assert False, "should not go here"
            for rec in ret:
                rec["question"] = "Please describe the mechanism of this drug."
                ans = rec.pop("abstract")
                rec["answer"] = ans

    print("total data size:", len(out))
    with open(f"{path_prefix}/dataset/uspto_QA_train.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == '__main__':
    path_prefix = '/home/zhangyu/drugchat/'
    # list_of_path = ["dataset/chembl_QA_train.pkl", "dataset/PubChem_QA_train.pkl"]
    list_of_path = ["dataset/uspto_QA_train.pkl"]
    main(list_of_path, path_prefix)
