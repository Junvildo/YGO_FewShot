import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import faiss
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data_preprocess import extract_artwork
from models import EmbeddedFeatureWrapper, GeM
from mobileone import reparameterize_model, mobileone
from data import InferenceDataset, CustomDataset
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLO model for object detection
yolo_model = YOLO('finetuned_models/yolo_ygo.pt')
id2name = json.load(open('id2name.json', 'r'))

# Load FAISS index and base dataset for similarity search
faiss_index_file = "precompute_embs/final/full_class_embeddings_224.faiss"
index = faiss.read_index(faiss_index_file)

# Load the feature extraction model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
feature_model.feature.gap = GeM()
state_dict = torch.load("precompute_embs/final/epoch_30_224.pth", map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
feature_model.load_state_dict(state_dict)

# Define transformations for input images
mean, std = [0.4935736358165741, 0.46013686060905457, 0.4618111848831177], [0.2947998642921448, 0.28370970487594604, 0.2891422510147095]
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Load base dataset for label-to-classname mapping
data_root = "dataset"
base_dataset = CustomDataset(root=data_root, train=True, transform=trans)

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
    dataset = InferenceDataset(numpy_arrays, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    model_eval = reparameterize_model(model)

    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch_outputs = model_eval(batch)
            outputs.append(batch_outputs.cpu())
    return torch.cat(outputs, dim=0)

def process_input(input_data, thresh_val):
    """
    Process input image using YOLO, perspective correction, and FAISS similarity search.
    Returns:
        - Annotated frame with bounding boxes and image names (or original frame if no cards are detected)
        - List of similar image names (or None if no cards are detected)
    """
    # Run YOLO inference
    results = yolo_model(input_data)
    cards = []

    # Extract bounding box coordinates and confidence scores
    for i, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):
        if conf >= 0.2:  # Filter by confidence (greater than 50%)
            x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
            cards.append(input_data[y1:y2, x1:x2])
    # print(f"Number of detected cards: {len(cards)}")

    # If no cards are detected, return the original frame and None for similar_image_names
    if not cards:
        return input_data, None

    # Preprocess cards (artwork extraction)
    arts = [extract_artwork(card, thresh_val=thresh_val, img_size=224) for card in cards]
    arts = [art for art in arts if art is not None]  # Filter out None values

    # If no valid artworks are extracted, return the original frame and None for similar_image_names
    if not arts:
        return input_data, None

    # Get model outputs for extracted artworks
    batch_size = 8 if len(arts) > 8 else len(arts)
    # print(f"Batch size: {batch_size}")
    outputs = get_model_outputs(arts, feature_model, batch_size, device, trans)
    outputs = outputs.detach().numpy()
    binary_query_embeddings = np.require(outputs > 0, dtype='float32')

    # Perform FAISS similarity search
    distances, indices = index.search(binary_query_embeddings, 1)
    k_similar_images = [indice for indice in indices]

    # Get image names of similar images
    similar_image_ids = []
    for i in range(len(k_similar_images)):
        image_name = base_dataset.label_to_classname[base_dataset.class_labels_list[k_similar_images[i][0]]]
        similar_image_ids.append(image_name)

    similar_image_names = [id2name[id] for id in similar_image_ids]

    # Render bounding boxes and image names on the input image
    annotated_frame = input_data.copy()
    for i, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):
        if conf >= 0.2:  # Filter by confidence (greater than 50%)
            x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Write image name on the bounding box
            if i < len(similar_image_names):
                image_name = similar_image_names[i]
            else:
                image_name = "Unknown"
            cv2.putText(annotated_frame, image_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return annotated_frame, similar_image_names

def gradio_interface(input_data, thresh_val):
    """
    Gradio interface for processing images.
    """
    # Process image
    annotated_frame, similar_image_names = process_input(input_data, thresh_val)
    if similar_image_names is None:
        return annotated_frame, "No cards detected or no valid artworks extracted."
    else:
        return annotated_frame, ", ".join(similar_image_names)

# Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# YOLO Object Detection with Gradio")
    
    # First row: Upload Image and Annotated Image
    with gr.Row():
        image_input = gr.Image(label="Upload Image")
        image_output = gr.Image(label="Annotated Image")
    
    # Second row: Slider
    with gr.Row():
        thresh_val_slider = gr.Slider(minimum=0, maximum=255, value=140, label="Threshold Value for Artwork Extraction")
    
    # Similar Image Names and Button
    similar_images_output = gr.Textbox(label="Similar Image Names")
    image_button = gr.Button("Detect Objects in Image")

    # Define button action
    image_button.click(
        fn=gradio_interface,
        inputs=[image_input, thresh_val_slider],
        outputs=[image_output, similar_images_output],
    )

# Launch the app
demo.launch()