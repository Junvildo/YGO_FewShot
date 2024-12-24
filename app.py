import cv2
from data_preprocess import detect_and_correct_perspective, extract_artwork
from ultralytics import YOLO
import numpy as np
import faiss
from models import EmbeddedFeatureWrapper
from mobileone import reparameterize_model, mobileone
import os
import torch
from torchvision import transforms
from data import InferenceDataset
from torch.utils.data import DataLoader
from PIL import Image
from data import CustomDataset


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


image_path = 'real_life_images/ygo_field2.jpg'

field_image = cv2.imread(image_path)

model = YOLO('finetuned_models/yolo_ygo.pt')

results = model(field_image)
cards = []

# Extract bounding box coordinates and confidence scores
for i, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):
    if conf >= 0.5:  # Filter by confidence (greater than 50%)
        x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
        # Crop the detected object from the image
        # cards.append(field_image[round(y1*0.98):round(y2*1.02), round(x1*0.98):round(x2*1.02)])
        cards.append(field_image[round(y1-2):round(y2+2), round(x1-2):round(x2+2)])

preprocessed_cards = []
for card in cards:
    preprocessed_cards.append(detect_and_correct_perspective(card))

arts = []
for card in preprocessed_cards:
    art = extract_artwork(card, thresh_val=180, img_size=56)
    if art is not None:
        arts.append(art)
    else:
        print("No artwork found in the image.")



# Concatenate all detected cards into one big image
resized_cards = []
for card in cards:
    resized_cards.append(cv2.resize(card, (56, 56)))
big_card_image = np.concatenate(resized_cards, axis=1)

arts_batch = []
for art in arts:
    arts_batch.append(art)
big_art_image = np.concatenate(arts_batch, axis=1)

# Display the big image
before_after = np.concatenate((big_card_image, big_art_image), axis=0)
cv2.imshow('Before and After', before_after)

def get_model_outputs(numpy_arrays, model, batch_size, device, transform):
    """
    Args:
        numpy_arrays: List of numpy arrays representing images.
        model: PyTorch model for inference.
        batch_size: Batch size for inference.
        device: Device to run inference on ('cpu' or 'cuda').
        transform: Transformations to be applied to each image.
    
    Returns:
        List of model outputs.
    """
    # Create the dataset and dataloader
    dataset = InferenceDataset(numpy_arrays, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Move model to the specified device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    model_eval = reparameterize_model(model)

    outputs = []
    with torch.no_grad():  # Disable gradient computation for inference
        for batch in dataloader:
            batch = batch.to(device)  # Move batch to device
            batch_outputs = model_eval(batch)  # Get model predictions
            outputs.append(batch_outputs.cpu())  # Store predictions, move to CPU

    # Concatenate all batches to get final output
    return torch.cat(outputs, dim=0)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
state_dict = torch.load("finetuned_models/s2_56_grayscale.pth", map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
mean, std = [0.39111483097076416, 0.38889095187187195, 0.38865992426872253], [0.30603259801864624, 0.306450754404068, 0.30432021617889404]
trans = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
data_root = "dataset"
base_dataset = CustomDataset(root=data_root, train=True, transform=trans)

base_loader = DataLoader(base_dataset,
                        batch_size=16,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0)
arts_image = [Image.fromarray(art) for art in arts]
outputs = get_model_outputs(arts, model, 8, device, trans)
outputs = outputs.detach().numpy()
binary_query_embeddings = np.require(outputs > 0, dtype='float32')
# Load the FAISS index and labels
faiss_index_file = "class_embeddings.faiss"
index = faiss.IndexFlatL2(2048)
index = faiss.read_index(faiss_index_file)


distances, indices = index.search(binary_query_embeddings, 4)

k_similar_images = [indice for indice in indices]

for i in range(len(k_similar_images)):
    print(str(i)+"\t", end="="*20+"\n")
    images_name = [base_dataset.label_to_classname[base_dataset.class_labels_list[j]] for j in k_similar_images[i]]
    images_path = ['https://images.ygoprodeck.com/images/cards_cropped/{}.jpg'.format(image_name) for image_name in images_name]
    for url in images_path:
        print(url)




cv2.waitKey(0)
cv2.destroyAllWindows()