# image mol
 torchrun --nproc_per_node 8 --master_port 34218 train.py --cfg-path train_configs/drugchat_image_mol.yaml
# python3 train.py --cfg-path train_configs/drugchat_image_mol.yaml
# image mol with wiki data
#torchrun --nproc_per_node 1 --master_port 34218 train.py --cfg-path train_configs/drugchat_image_mol_wiki.yaml