import torch
import faiss
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from models import EmbeddedFeatureWrapper
from mobileone import mobileone, reparameterize_model
import time
from data import CustomDataset, ImageDataset
import json
from extract_features import extract_feature
import requests
import os


# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
state_dict = torch.load("finetuned_models/s2_224_color_resize.pth", map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
model_eval = reparameterize_model(model)

# Transform setup
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Data setup
image_dir = "dataset_artworks_all"
base_dataset = CustomDataset(root=image_dir, train=True, transform=trans)

base_loader = DataLoader(base_dataset,
                        batch_size=8,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0)
label2id = base_dataset.label_to_classname
card_data = requests.get("https://db.ygoprodeck.com/api/v7/cardinfo.php").json()
id2name = {card_data['data'][i]['id']: card_data['data'][i]['name'] for i in range(len(card_data['data']))}
label2name = {label: id2name[int(idx)] for label, idx in label2id.items()}
with open("label2name.json", "w") as outfile:
    json.dump(label2name, outfile)
start = time.time()
embeddings, labels = extract_feature(loader=base_loader, model=model_eval, device=device)
binary_db_embeddings = np.require(embeddings > 0, dtype='float32')

# Index setup
index = faiss.IndexFlatL2(binary_db_embeddings.shape[1])
index.add(binary_db_embeddings)

# Write index to disk
faiss.write_index(index, "class_embeddings.faiss")

end = time.time()
print("Time cost: {} seconds".format(end - start))