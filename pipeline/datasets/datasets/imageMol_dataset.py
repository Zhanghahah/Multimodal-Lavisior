import os
import json
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms



class ImageMolDataset(Dataset):
    def __init__(self, datapath, image_size=224) -> None:
        super().__init__()
        jsonpath = os.path.join(datapath, "img_qa.json")
        print(f"Using {jsonpath=}")
        with open(jsonpath, "rt") as f:
            meta = json.load(f)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])


        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size), 
            transforms.ToTensor(),
            normalize,
        ])
        self.images = {}
        self.data = []
        for idx, (img_key, rec) in enumerate(tqdm(meta.items())):  # for debug
            img_file = img_key
            image_folder_list = os.path.join(datapath, img_file)
            img_list = []
            for image_path in os.listdir(image_folder_list):
                image = Image.open(os.path.join(image_folder_list, image_path)).convert("RGB")
                img = self.transforms(image)
                img_list.append(img)
            # full_img = torch.cat(img_list, 0)
            self.images[idx] = img
            self.data.append([idx, rec])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        idx, qa_pair = self.data[index]
        img = self.images[idx]
        return {"img": img, "question": qa_pair[0], "text_input": str(qa_pair[1])}
    
    @staticmethod
    def collater(samples):
        imgs = default_collate([x["img"] for x in samples])
        qq = [x["question"] for x in samples]
        aa = [x["text_input"] for x in samples]
        out = {"image": imgs, "question": qq, "text_input": aa}
        return out
