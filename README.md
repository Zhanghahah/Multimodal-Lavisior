# Multimodal LLM for multi tasks: Towards AI for Chemistry

This repository holds the code and data of Lavoisier: Towards specific chemical tasks: Forward Prediction, Retrosynthesis, Condition Generation, Yield Prediction.


## Examples




## Introduction
- In this work, we make an initial attempt towards enabling ChatGPT-like capabilities on drug molecule graphs, by developing a prototype multimodal chemical model
- The whole system consists of a graph neural network (GNN), a large language model (LLM), and an project layer. The GNN takes a compound molecule graph or a reaction reresentation as input and learns a representation for this graph. The project layer transforms the graph representation produced by the GNN  into another  representation that is acceptable to the  LLM. The LLM takes the compound representation transformed by the adaptor and users' questions about this reaction as inputs and generates answers. All these steps are trained end-to-end.
- To train multimodal Lavoisier, we collected self-instruction tuning datasets from USPTO-50k.



## Datasets 

The file `dataset/IQA.py` contains cooked data for Supervised Instruction Tuning DatasetThe data structure is as follows. 

- {SMILES String: [ [Question1 , Answer1], [Question2 , Answer2]... ] }
- {<image> [Task descritor, answer]}

## GNN Pipeline Clarification

**1. datasets**
- cook graph data 
```
dataset/uspto_smiles2graph.py: 
    input: smiles
    output: {"graph": graph, "question": question, "answer": str(answer)}
```
```
dataset/combine_convert_datasets:
    input:{"graph": graph, "question": question, "answer": str(answer)}
    output: endow graph to Data() in dgl ex:graph = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
```

```

pipeline/datasets/datasets/cc_sbu_dataset.py
     graph = rec["graph"]
     caption = rec["answer"]
     question = rec["question"]
```
- training_loop

```
pipeline/models/mini_gpt4.py: the input of self.encode_img includes graph, answer, question 
img_embeds, atts_img = self.encode_img(samples, device)

```
## Getting Started
### Installation
These instructions largely follow those in MiniGPT-4.

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and ativate it via the following command

```bash
git clone https://github.com/Zhanghahah/Multimodal-Lavisior.git
cd Multimodal-Lavisior
conda env create -f environment.yml
```
check the version of torch which need to meet `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`


Verify the installation of `torch` and `torchvision` is successful by running `python -c "import torchvision; print(torchvision.__version__)"`. If it outputs the version number without any warnings or errors, then you can go to the next step (installing PyTorch Geometric). __If it outputs any warnings or errors__, try to uninstall `torch` by `conda uninstall pytorch torchvision torchaudio cudatoolkit` and then reinstall them following [here](https://pytorch.org/get-started/previous-versions/#v1121). You need to find the correct command according to the CUDA version your GPU driver supports (check `nvidia-smi`). For example, I found my GPU driver supported CUDA 11.6, so I run `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`.


**2. Prepare the pretrained Vicuna weights**

The current version of DrugChat is built on the v0 versoin of Vicuna-13B.
Please refer to our instruction [here](PrepareVicuna.md) 
to prepare the Vicuna weights.
The final weights would be in a single folder in a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

Then, set the path to the vicuna weight in the model config file 
[here](pipeline/configs/models/drugchat.yaml#L16) at Line 16.

### Training
**You need roughly 40 GB GPU memory for the training.** 

The training configuration file is [train_configs/drugchat_stage2_finetune.yaml](train_configs/drugchat_stage2_finetune.yaml). You may want to change the number of epochs and other hyper-parameters there, such as `max_epoch`, `init_lr`, `min_lr`,`warmup_steps`, `batch_size_train`. You need to adjust `iters_per_epoch` so that `iters_per_epoch` * `batch_size_train` = your training set size.

Start training the projection layer that connects the GNN output and the LLaMA model by running `sh finetune_gnn.sh`. 

### Inference 
#### IQA task
**start infer, first find the trained ckpt like**
```
pipeline/output/pipeline_image_mol/20231024181/checkpoint_1.pt
```

**modify file inference configuration in line 11 [eval_configs/image_mol_infer.yaml](eval_configs/image_mol_infer.yaml), then run `sh inference_imgMol.sh`**

#### Lavoisier
**To get the inference to work properly, you need to create another environment (`rdkit`) and launch a backend process which converts SMILES strings to Torch Geometric graphs.**

**It takes around 24 GB GPU memory for the demo.**

To create the `rdkit` environment and run the process, run
```
conda create -c conda-forge -n rdkit rdkit
conda activate rdkit
pip install numpy
python dataset/smiles2graph_demo.py
```
Then, the `smiles2graph_demo.py` will be running in the backend to serve the `demo.py`.

Find the checkpoint you save in the training process above, which is located under the folder `pipeline/output/pipeline_stage2_finetune/` by default. Copy it to the folder `ckpt` by running `cp pipeline/output/pipeline_stage2_finetune/the_remaining_path ckpt/with_gnn_node_feat.pth`. 

Now we launch the `demo.py` in our original environment. Make sure you have run `conda activate drugchat`. Then, start the demo [demo.sh](demo.sh) on your local machine by running `bash demo.sh`. Then, open the URL created by the demo and try it out!


## Acknowledgement

+ [MiniGPT-4](https://minigpt-4.github.io/) This repo is based on MiniGPT-4, an awesome repo for vision-language chatbot!
+ [Lavis](https://github.com/salesforce/LAVIS)
+ [Vicuna](https://github.com/lm-sys/FastChat)


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) with BSD 3-Clause License [here](LICENSE_MiniGPT4.md), which is based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).


## Disclaimer

This is a prototype system that has not been systematically and comprehensively validated by pharmaceutical experts yet. Please use with caution. 

Trained models and demo websites will be released after we thoroughly validate the system with pharmaceutical experts.


## Citation

