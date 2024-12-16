import os
import torch
import faiss
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from models import EmbeddedFeatureWrapper
from mobileone import mobileone, reparameterize_model
from data import ImageDataset
from util import calculate_mean_std

# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
state_dict = torch.load("./finetuned_models/s2_56_epoch_45_newest.pth", map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
# model_eval = reparameterize_model(model)

# Transform setup
# mean, std = calculate_mean_std("dataset_artworks_training", num_workers=0)
mean = [0.49617722630500793, 0.46303924918174744, 0.46300116181373596]
std = [0.1427406221628189, 0.13628196716308594, 0.13478131592273712]
trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Helper function to binarize embeddings
def binarize_embeddings(embeddings):
    return (embeddings > 0)

# Paths and labels setup
image_dir = "dataset_artworks"
class_to_images = defaultdict(list)
labels = []  # Fill this with class labels for each image path
image_paths = []  # Fill this with the corresponding image paths


# Populate image paths and labels
for root, _, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust for valid extensions
            full_path = os.path.join(root, file)
            image_paths.append(full_path)
            labels.append(os.path.basename(root))

print(f"Number of images: {len(image_paths)}")
print(f"Sample paths: {image_paths[:5]}")
print(f"Sample labels: {labels[:5]}")

# Map images to their classes
for image_path, label in zip(image_paths, labels):
    class_to_images[label].append(image_path)

# Compute average embeddings for each class
class_embeddings = {}
for class_label, class_images in class_to_images.items():
    dataset = ImageDataset(class_images, [class_label] * len(class_images), trans)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            features = model(images).cpu().numpy()
            embeddings.append(features)
    
    if embeddings:
        embeddings = np.vstack(embeddings)
        average_embedding = np.mean(embeddings, axis=0)
        class_embeddings[class_label] = binarize_embeddings(average_embedding)

# Save embeddings and labels to FAISS index
dimension = 2048  # Update this according to your model's output
index = faiss.IndexFlatL2(dimension)
labels = []

for class_label, embedding in class_embeddings.items():
    index.add(embedding.reshape(1, -1))
    labels.append(class_label)

# Write index to disk
faiss.write_index(index, "class_embeddings.faiss")

# Save labels
with open("labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

# # FAISS search
# def search_image(image_path):
#     dataset = ImageDataset([image_path], ["query"], trans)
#     dataloader = DataLoader(dataset, batch_size=1)
    
#     with torch.no_grad():
#         for images, _ in dataloader:
#             images = images.to(device)
#             query_embedding = model_eval(images).cpu().numpy()
#             query_embedding = binarize_embeddings(query_embedding)
    
#     index = faiss.read_index_binary("class_embeddings.faiss")
#     _, indices = index.search(query_embedding, 1)
    
#     with open("labels.txt", "r") as f:
#         labels = f.read().splitlines()
    
#     return labels[indices[0][0]]

# # Example search
# result = search_image("real_life_images/detected_object_1.jpg")
# print(f"Closest label: {result}")
