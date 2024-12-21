from extract_features import extract_feature
import os
import torch
import faiss
from torchvision import transforms
from models import EmbeddedFeatureWrapper
from mobileone import mobileone, reparameterize_model
from torch.utils.data import DataLoader
from data import CustomDataset
import json
from PIL import Image
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
state_dict = torch.load("finetuned_models/s2_56_best_color.pth", map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
model_eval = reparameterize_model(model)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
trans = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

data_root = "dataset_artworks_all"
base_dataset = CustomDataset(root=data_root, train=True, transform=trans)

base_loader = DataLoader(base_dataset,
                        batch_size=32,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0)

label2name = json.load(open("label2name.json", "r"))

img_path = "real_life_images/detected_object_2.jpg"
img = Image.open(img_path).convert('RGB')
query_embedding = model_eval(torch.unsqueeze(trans(img), 0).to(device))
if os.path.isfile("class_embeddings.faiss"):    
    index = faiss.IndexFlatL2(2048)  # Euclidean distance index
    index = faiss.read_index("class_embeddings.faiss")
else:
    embeddings, labels = extract_feature(loader=base_loader, model=model_eval, device=device)
    binary_db_embeddings = np.require(embeddings > 0, dtype='float32')
    index = faiss.IndexFlatL2(2048)  # Euclidean distance index
    index.add(binary_db_embeddings)
    faiss.write_index(index, "class_embeddings.faiss")

binary_query_embeddings = np.require(query_embedding > 0, dtype='float32')

distances, indices = index.search(query_embedding, 1)


image_names = [base_dataset.image_paths[i] for i in indices[0]]
image_classes = [base_dataset.class_labels[i] for i in indices[0]]

print(image_names)
print(image_classes)