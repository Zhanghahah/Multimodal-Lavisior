"""
Date: 2023/08/24
Author: cynthiazhang
smiles2graph for uspto datasets
"""

from rdkit import Chem
import numpy as np
import json
import pickle
from tqdm import tqdm
from rdkit.Chem.rdchem import BondType, BondDir, ChiralType

BOND_TYPE = {
    BondType.SINGLE: 0, BondType.DOUBLE: 1,
    BondType.TRIPLE: 2, BondType.AROMATIC: 3
}
BOND_DIR = {
    BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2
}
CHI = {
    ChiralType.CHI_UNSPECIFIED: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2, ChiralType.CHI_OTHER: 3
}


def bond_dir(bond):
    d = bond.GetBondDir()
    return BOND_DIR[d]


def bond_type(bond):
    t = bond.GetBondType()
    return BOND_TYPE[t]


def atom_chiral(atom):
    c = atom.GetChiralTag()
    return CHI[c]


def atom_to_feature(atom):
    return [atom.GetAtomicNum() - 1, atom_chiral(atom)]


def bond_to_feature(bond):
    return [bond_type(bond), bond_dir(bond)]


def canonical_smiles(smi):
    """
        Canonicalize a SMILES without atom mapping
        """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    else:
        canonical_smi = Chem.MolToSmiles(mol)
        # print('>>', canonical_smi)
        if '.' in canonical_smi:
            canonical_smi_list = canonical_smi.split('.')
            canonical_smi_list = sorted(
                canonical_smi_list, key=lambda x: (len(x), x)
            )
            canonical_smi = '.'.join(canonical_smi_list)
        return canonical_smi


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 2
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in
        # COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with
        # shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def parse_json(path):
    with open(path) as r:
        data = r.readlines()
    data_length = len(data)
    meta_data = []
    for i in range(data_length):
        meta_data.append(json.loads(data[i]))
    return meta_data


def convert_uspto(path):
    # with open("/home/youwei/project/drugchat/data/ChEMBL_QA_train.json", "rt") as f:
    raw_data = parse_json(path)
    out = []
    total_length = len(raw_data)
    for i in tqdm(range(0, total_length)):
        data = raw_data[i]
        if data['type'] in ['reactants', 'products']:
            # smiles for reactants, products, or reagents
            smiles = data['input']
            question = data['instruction']
            answer = data['output']
            graph = smiles2graph(smiles)

            out.append({
                "graph": graph,
                "question": question,
                "answer": str(answer)
            })
        elif data['type'] in ['classification', 'yield']:
            reac, reag, prod = data['input'].split('>>')
            out.append({
                'reactants': smiles2graph(reac),
                'products': smiles2graph(prod),
                'reagents': smiles2graph(reag),
                'question': data['instruction'],
                'answer': str(data['output'])
            })
        elif data['type'] == 'reagents':
            reac, prod = data['input'].split('>>')
            out.append({
                'reactants': smiles2graph(reac),
                'products': smiles2graph(prod),
                'question': data['instruction'],
                'answer': str(data['output'])
            })
        else:
            raise NotImplementedError(f'Invalid type {data["type"]}')

    with open("/home/zhangyu/drugchat/dataset/uspto_QA_retrosynthesis_test.pkl", "wb") as f:
        pickle.dump(out, f)


def convert_simple_graph_smi(infile, outfile):
    """
    Convert to a data format that support both graph and images (which is converted to feature dataset later)
    """
    with open(infile, "rt") as f:
        js = json.load(f)
    out = {}
    for smi, rec in js.items():
        graph = smiles2graph(smi)
        out[smi] = {"graph": graph, "QA": rec}

    with open(outfile, "wb") as f:
        pickle.dump(out, f)


if __name__ == '__main__':
    # graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    # print(graph)
    path = "/home/zhangyu/data/uspto/chem_uspto_instruction_test_v2.json"
    convert_uspto(path)
    # convert_simple_graph_smi("data/ChEMBL_PubChem_QA_train.json", "dataset/ChEMBL_PubChem_QA_train_graph_smi.pkl")
    # convert_simple_graph_smi("data/ChEMBL_QA_train.json", "dataset/ChEMBL_QA_train_graph_smi.pkl")
    # convert_simple_graph_smi("data/ChEMBL_QA_test.json", "dataset/ChEMBL_QA_test_graph_smi.pkl")
