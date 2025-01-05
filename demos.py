import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# Load the YOLO model
model = YOLO("finetuned_models/yolo_ygo.pt")

# Function to process image/video/webcam input
def process_input(input_type, input_data):
    if input_type == "webcam":
        # Read frame from webcam
        frame = input_data
    elif input_type == "image":
        # Read image
        frame = input_data
    elif input_type == "video":
        # Read video frame by frame
        frame = input_data
    else:
        raise ValueError("Invalid input type")

    # Run YOLO inference
    results = model(frame)

    # Render bounding boxes on the frame
    annotated_frame = results[0].plot()

    return annotated_frame

# Gradio interface
def gradio_interface(input_type, input_data):
    if input_type == "image":
        # Process image
        output_frame = process_input(input_type, input_data)
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
        annotated_frames = [process_input(input_type, frame) if i % 30 == 0 else frame for i, frame in enumerate(frames)]
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
        return output_video.name
    else:
        return "Invalid input type"

    return output_frame

# Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# YOLO Object Detection with Gradio")
    with gr.Tab("Image"):
        image_input = gr.Image(label="Upload Image")
        image_output = gr.Image(label="Image Output")
        image_button = gr.Button("Detect Objects in Image")
    with gr.Tab("Video"):
        video_input = gr.Video(label="Upload Video")
        video_output = gr.Video(label="Video Output")
        video_button = gr.Button("Detect Objects in Video")

    # Define button actions
    image_button.click(
        fn=gradio_interface,
        inputs=[gr.State("image"), image_input],
        outputs=image_output,
    )
    video_button.click(
        fn=gradio_interface,
        inputs=[gr.State("video"), video_input],
        outputs=video_output,
    )

# Launch the app
demo.launch()