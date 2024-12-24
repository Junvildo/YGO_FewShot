import torch
import faiss
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from models import EmbeddedFeatureWrapper
from mobileone import mobileone, reparameterize_model
import time
from data import CustomDataset
from extract_features import extract_feature


# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
state_dict = torch.load("finetuned_models/s2_56_grayscale.pth", map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
model_eval = reparameterize_model(model)

# Transform setup
mean, std = [0.39111483097076416, 0.38889095187187195, 0.38865992426872253], [0.30603259801864624, 0.306450754404068, 0.30432021617889404]
trans = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Data setup
image_dir = "dataset"
base_dataset = CustomDataset(root=image_dir, train=True, transform=trans)

base_loader = DataLoader(base_dataset,
                        batch_size=32,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0)
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
