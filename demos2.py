import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import faiss
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from data_preprocess import detect_and_correct_perspective, extract_artwork
from models import EmbeddedFeatureWrapper
from mobileone import reparameterize_model, mobileone
from data import InferenceDataset, CustomDataset
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLO model for object detection
yolo_model = YOLO('finetuned_models/yolo_ygo.pt')

# Load FAISS index and base dataset for similarity search
faiss_index_file = "class_embeddings.faiss"
index = faiss.read_index(faiss_index_file)

# Load the feature extraction model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
state_dict = torch.load("finetuned_models/s2_56_grayscale.pth", map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
feature_model.load_state_dict(state_dict)

# Define transformations for input images
mean, std = [0.49362021684646606, 0.4601792097091675, 0.4618436098098755], [0.27437326312065125, 0.2629182040691376, 0.270280659198761]
trans = transforms.Compose([
    transforms.Resize((56, 56)),
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

def process_input(input_data):
    """
    Process input image or video using YOLO, perspective correction, and FAISS similarity search.
    Returns:
        - Annotated frame with bounding boxes
        - Concatenated image of all detected cards
        - Concatenated image of all extracted artworks
        - List of similar image URLs
    """
    # Run YOLO inference
    results = yolo_model(input_data)
    cards = []

    # Extract bounding box coordinates and confidence scores
    for i, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):
        if conf >= 0.5:  # Filter by confidence (greater than 50%)
            x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
            cards.append(input_data[round(y1-2):round(y2+2), round(x1-2):round(x2+2)])

    # Preprocess cards (perspective correction and artwork extraction)
    preprocessed_cards = [detect_and_correct_perspective(card) for card in cards]
    arts = [extract_artwork(card, thresh_val=180, img_size=56) for card in preprocessed_cards]
    arts = [art for art in arts if art is not None]  # Filter out None values

    # Get model outputs for extracted artworks
    outputs = get_model_outputs(arts, feature_model, 8, device, trans)
    outputs = outputs.detach().numpy()
    binary_query_embeddings = np.require(outputs > 0, dtype='float32')

    # Perform FAISS similarity search
    distances, indices = index.search(binary_query_embeddings, 1)
    k_similar_images = [indice for indice in indices]

    # Get URLs of similar images
    similar_image_urls = []
    for i in range(len(k_similar_images)):
        images_name = [base_dataset.label_to_classname[base_dataset.class_labels_list[j]] for j in k_similar_images[i]]
        images_path = ['https://images.ygoprodeck.com/images/cards_cropped/{}.jpg'.format(image_name) for image_name in images_name]
        similar_image_urls.extend(images_path)

    # Render bounding boxes on the input image
    annotated_frame = results[0].plot()

    # Stack detected cards into one image
    if len(cards) > 0:
        resized_cards = [cv2.resize(card, (100, 100)) for card in cards]  # Resize cards to a fixed size
        concatenated_cards = np.hstack(resized_cards)  # Stack cards horizontally
    else:
        concatenated_cards = np.zeros((100, 100, 3), dtype=np.uint8)  # Return a blank image if no cards are detected

    # Stack extracted artworks into one image
    if len(arts) > 0:
        resized_arts = [cv2.resize(art, (100, 100)) for art in arts]  # Resize artworks to a fixed size
        concatenated_arts = np.hstack(resized_arts)  # Stack artworks horizontally
    else:
        concatenated_arts = np.zeros((100, 100, 3), dtype=np.uint8)  # Return a blank image if no artworks are extracted

    return annotated_frame, concatenated_cards, concatenated_arts, similar_image_urls

# Gradio interface
def gradio_interface(input_type, input_data):
    if input_type == "image":
        # Process image
        annotated_frame, concatenated_cards, concatenated_arts, similar_image_urls = process_input(input_data)
        return annotated_frame, concatenated_cards, concatenated_arts, similar_image_urls
    elif input_type == "video":
        # Process video
        cap = cv2.VideoCapture(input_data)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        # Process each frame
        annotated_frames = [process_input(frame)[0] if i % 30 == 0 else frame for i, frame in enumerate(frames)]
        # Save annotated frames as a video
        output_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out = cv2.VideoWriter(
            output_video.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (annotated_frames[0].shape[1], annotated_frames[0].shape[0]),
        )
        for frame in annotated_frames:
            out.write(frame)
        out.release()
        return output_video.name, np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8), []  # Return blank images for video
    else:
        return "Invalid input type", np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8), []

# Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# YOLO Object Detection with Gradio")
    with gr.Tab("Image"):
        image_input = gr.Image(label="Upload Image")
        image_output = gr.Image(label="Annotated Image")
        concatenated_cards_output = gr.Image(label="Detected Cards")
        concatenated_arts_output = gr.Image(label="Extracted Artworks")
        similar_images_output = gr.Textbox(label="Similar Image URLs")
        image_button = gr.Button("Detect Objects in Image")
    with gr.Tab("Video"):
        video_input = gr.Video(label="Upload Video")
        video_output = gr.Video(label="Annotated Video")
        video_button = gr.Button("Detect Objects in Video")

    # Define button actions
    image_button.click(
        fn=gradio_interface,
        inputs=[gr.State("image"), image_input],
        outputs=[image_output, concatenated_cards_output, concatenated_arts_output, similar_images_output],
    )
    video_button.click(
        fn=gradio_interface,
        inputs=[gr.State("video"), video_input],
        outputs=[video_output, concatenated_cards_output, concatenated_arts_output, similar_images_output],
    )

# Launch the app
demo.launch()