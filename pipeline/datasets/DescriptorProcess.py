# -*- coding: utf-8 -*-
"""
@Author: Cynthia Zhang
@Affiliation: SJTU
@Date: 2023/10/20
"""
import os
import json
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from dscribe.descriptors import MBTR, SOAP, LMBTR

descs = [desc_name[0] for desc_name in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)


def descriptor_calculation(data_with_task):
    """this function now server for reactants representation"""
    all_reactants = set()
    for _idx, task_data in enumerate(data_with_task):
        mol_smiles = task_data['input']  # smiles for reactants, products, or reagents
        for mol_smi in mol_smiles.split('.'):
            all_reactants.add(mol_smi)
        question = task_data['instruction']
        answer = task_data['output']
    all_reactants = list(all_reactants)
    for reactant in all_reactants:
        #import pudb
        #pudb.set_trace()
        #os.system("echo {reactant} > {reactant}.txt")
        cmd_xyz = f"echo '{reactant}' | obabel -i smi -o xyz -O '{reactant}.xyz' --gen3d"
        os.system(cmd_xyz)
        cmd_sdf = f"obabel '{reactant}.xyz' -O '{reactant}.sdf' --gen3d --conformer --nconf 20 --weighted"
        os.system(cmd_sdf)

def get_atom_species(smiles_set, smi_mol_map):
    species = []
    for smi in smiles_set:
        species += [atom.GetSymbol() for atom in smi_mol_map[smi].GetAtoms()]
    return list(set(species))


def maxminscale(array):
    '''
    Max-min scaler
    Parameters
    ----------
    array : ndarray
        Original numpy array.
    Returns
    -------
    array : ndarray
        numpy array with max-min scaled.
    '''
    return (array - array.min(axis=0)) / (array.max(axis=0) - array.min(axis=0))


def process_desc(array):
    array = np.array(array, dtype=np.float32)
    desc_len = array.shape[1]
    rig_idx = []
    for i in range(desc_len):
        try:
            desc_range = array[:, i].max() - array[:, i].min()
            if desc_range != 0 and not np.isnan(desc_range):
                rig_idx.append(i)
        except:
            continue
    array = array[:, rig_idx]
    return array


def genOneHot(compound_set):
    oh_map = {}
    for i, item in enumerate(compound_set):
        oh = [0] * len(compound_set)
        oh[i] = 1
        oh_map[item] = oh
    return oh_map


def getmorganfp(mol, radius=2, nBits=2048, useChirality=True):
    '''
    
    Parameters
    ----------
    mol : mol
        RDKit mol object.
    Returns
    -------
    mf_desc_map : ndarray
        ndarray of molecular fingerprint descriptors.
    '''
    fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
    return np.array(list(map(eval, list(fp.ToBitString()))))


def calc_rdkit_desc(mol):
    return desc_calc.CalcDescriptors(mol)


def calc_Dscribe_Desc(smi, smi_atoms_map, species, parameter_dict, type_='MBTR'):
    type_ = type_.lower()

    if type_ == 'mbtr':
        k1 = parameter_dict['k1']
        k2 = parameter_dict['k2']
        k3 = parameter_dict['k3']
        periodic = parameter_dict['periodic']
        normalization = parameter_dict['normalization']
        calculator = MBTR(species=species, k1=k1, k2=k2, k3=k3, periodic=periodic, normalization=normalization)
        return calculator.create(smi_atoms_map[smi])

    elif type_ == 'soap':
        rcut = parameter_dict['rcut']
        nmax = parameter_dict['nmax']
        lmax = parameter_dict['lmax']
        periodic = parameter_dict['periodic']
        calculator = SOAP(species=species, periodic=periodic, rcut=rcut, nmax=nmax, lmax=lmax)
        return np.mean(calculator.create(smi_atoms_map[smi]), axis=0)

    elif type_ == 'lmbtr':
        k2 = parameter_dict['k2']
        k3 = parameter_dict['k3']
        periodic = parameter_dict['periodic']
        normalization = parameter_dict['normalization']
        calculator = LMBTR(species=species, k2=k2, k3=k3, periodic=False, normalization=normalization)
        return np.mean(calculator.create(smi_atoms_map[smi]), axis=0)


def box_cox_trans(x, lambda_):
    '''
    Box-Cox Transformation

    Parameters
    ----------
    x : ndarray
        DESCRIPTION.
    lambda_ : float
        DESCRIPTION.

    Returns
    -------
    ndarray
        DESCRIPTION.

    '''
    if lambda_ != 0:
        return (np.power(x, lambda_) - 1) / lambda_
    else:
        return np.log(x)


def de_box_cox_trans(x, lambda_):
    if lambda_ != 0:
        return np.power((1 + lambda_ * x), 1 / lambda_)
    else:
        return np.exp(np.power(x, lambda_))


def log_trans(x):
    '''
    Logarithmic Transformation

    Parameters
    ----------
    x : ndarray
        DESCRIPTION.

    Returns
    -------
    ndarray
        DESCRIPTION.

    '''

    return np.log((1 - x) / (1 + x))


def de_log_trans(x):
    return (1 - np.exp(x)) / (1 + np.exp(x))


def ee2ddG(ee, T):
    '''
    Transformation from ee to ΔΔG
    Parameters
    ----------
    ee : ndarray
        Enantiomeric excess.
    T : ndarray or float
        Temperature (K).

    Returns
    -------
    ddG : ndarray
        ΔΔG (kcal/mol).
    '''

    ddG = np.abs(8.314 * T * np.log((1 - ee) / (1 + ee)))  # J/mol
    ddG = ddG / 1000 / 4.18  # kcal/mol
    return ddG


def ddG2ee(ddG, T):
    '''
    Transformation from ΔΔG to ee. 
    Parameters
    ----------
    ddG : ndarray
        ΔΔG (kcal/mol).
    T : ndarray or float
        Temperature (K).

    Returns
    -------
    ee : ndarray
        Absolute value of enantiomeric excess.
    '''

    ddG = ddG * 1000 * 4.18
    ee = (1 - np.exp(ddG / (8.314 * T))) / (1 + np.exp(ddG / (8.314 * T)))
    return np.abs(ee)


def parse_json(path):
    with open(path) as r:
        data = r.readlines()
    data_length = len(data)
    meta_data = []
    for i in range(data_length):
        meta_data.append(json.loads(data[i]))
    return meta_data


if __name__ == '__main__':
    in_file = '/data/zhangyu/own_data/uspto/chem_uspto_instruction_test_v2.json'
    test_samples = parse_json(in_file)
    out = {}
    total_length = len(test_samples)
    total_embs = []
    data_for_tasks = []
    for idx in tqdm(range(0, total_length)):
        data = test_samples[idx]
        if data['type'] == 'products':
            data_for_tasks.append(data)
    assert len(data_for_tasks) < total_length

    descriptor_calculation(data_for_tasks)
