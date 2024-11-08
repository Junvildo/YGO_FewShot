# from mobileone import mobileone
# from torchsummary import summary
# from torch import nn

# class EmbeddedFeatureWrapper(nn.Module):
#     """
#     Wraps a base model with embedding layer modifications.
#     """
#     def __init__(self,
#                  feature,
#                  input_dim,
#                  output_dim):
#         super(EmbeddedFeatureWrapper, self).__init__()

#         self.feature = nn.Sequential(
#             feature.stage0,
#             feature.stage1,
#             feature.stage2,
#             feature.stage3,
#             feature.stage4,
#             feature.gap
#         )
#         self.standardize = nn.LayerNorm(input_dim, elementwise_affine=False)

#         self.remap = None
#         if input_dim != output_dim:
#             self.remap = nn.Linear(input_dim, output_dim, bias=False)

#     def forward(self, images):
#         x = self.feature(images)
#         x = x.view(x.size(0), -1)
#         x = self.standardize(x)

#         if self.remap:
#             x = self.remap(x)

#         x = nn.functional.normalize(x, dim=1)

#         return x

#     def __str__(self):
#         return "{}_{}".format(self.feature.name, str(self.embed))


# baseline = mobileone(variant="s2")
# model = EmbeddedFeatureWrapper(feature=baseline, input_dim=2048, output_dim=2048)

# model = mobileone(variant="s2")
# summary(model, (3, 7, 7))

# from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import default_collate
# from torchvision import transforms
# from sampler import ClassBalancedBatchSampler
# from data import CustomDataset


# train_transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# train_dataset = CustomDataset(root="./main_dataset", train=True, transform=train_transform)

# train_sampler = ClassBalancedBatchSampler(train_dataset.class_labels_list, 2, 2)
# train_loader = DataLoader(train_dataset,
#                             batch_sampler=train_sampler, num_workers=4,
#                             pin_memory=True, drop_last=False, collate_fn=default_collate)

# for data in train_loader:
#     matrix, class_labels = data
#     print(matrix.shape, class_labels.shape)
#     break


# train_loader = DataLoader(train_dataset,
#                             num_workers=4, batch_size=2,
#                             pin_memory=True, drop_last=False, collate_fn=default_collate)

# for data in train_loader:
#     matrix, class_labels = data
#     print(matrix.shape, class_labels.shape)
#     break

# import faiss
# import numpy as np
# from models import EmbeddedFeatureWrapper
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from data import CustomDataset
# from torch.utils.data.dataloader import default_collate
# from extract_features import extract_feature
# from mobileone import mobileone
# import torch
# import os
# import time

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from PIL import Image

# def create_query_embedding(image_path, model, device):
#     # Load and preprocess the image
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.Resize((56, 56)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     # Evaluate the model to get the embedding
#     model.eval()
#     with torch.no_grad():
#         query_embedding = model(image_tensor)

#     return query_embedding.cpu().numpy()


# def retrieve_knn_faiss_euclidean(query_embeddings, db_embeddings, k):
#     """
#         Retrieve k nearest neighbors based on Euclidean distance using CPU.

#         Args:
#             query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
#             db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
#             k:                          number of nn results to retrieve excluding query

#         Returns:
#             dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
#                                         for each query
#             retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
#                                         for each query
#     """
#     index = faiss.IndexFlatL2(db_embeddings.shape[1])  # Euclidean distance index
#     index.add(db_embeddings)
#     dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

#     return dists, retrieved_result_indices

# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
#     state_dict = torch.load("./finetuned_models/s2_56_epoch_45.pth", map_location=device, weights_only=True)
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     model.load_state_dict(state_dict)
#     trans = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.Resize((56, 56)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     dataset = CustomDataset("D:/Coding/ygo_fsl/dataset/", transform=trans, train=True)
#     if not os.path.exists("embeddings.npy"):

#         dataloader = DataLoader(dataset, batch_size=8, num_workers=4,pin_memory=True, collate_fn=default_collate)
#         embeddings, labels = extract_feature(loader=dataloader, model=model, device=device)
#         # Save embeddings to a file
#         np.save("embeddings.npy", embeddings)

#     embeddings = np.load("embeddings.npy")
#     binary_db_embeddings = np.require(embeddings > 0, dtype='float32')


#     # Example usage
#     time_start = time.time()
#     image_path = "detected_object_3.jpg"
#     query_embedding = create_query_embedding(image_path, model, device)

#     binary_query_embeddings = np.require(query_embedding > 0, dtype='float32')


#     dists, retrieved_result_indices = retrieve_knn_faiss_euclidean(binary_query_embeddings, binary_db_embeddings, 1)
#     print(dists)
#     print(retrieved_result_indices)
#     # Convert retrieved indices to image names or image classes
#     image_names = [dataset.image_paths[i] for i in retrieved_result_indices[0]]
#     image_classes = [dataset.class_labels[i] for i in retrieved_result_indices[0]]

#     time_end = time.time()
#     print("Time cost: {} seconds".format(time_end - time_start))

#     print("Retrieved Image Names:", image_names)
#     print("Retrieved Image Classes:", image_classes)



# import json
# import requests


# with open('all_cards.json') as f:
#     data = json.load(f)
# data = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php').json()

# cards = [card for card in data['data'] if card['type'] not in ['Token', 'Skill Card']]

# Card image_url list
# url_list = [img['image_url_cropped'] for card in cards for img in card['card_images']]

# card_url_dict = {}
# for card in cards:
#     if len(card['card_images']) != 1:
#         card_url_dict[card['id']] = [url['image_url_cropped'] for url in card['card_images']]
#     else:
#         card_url_dict[card['id']] = [card['card_images'][0]['image_url_cropped']]



# with open('card_url_dict.json', 'w') as f:
#     json.dump(card_url_dict, f)

from ultralytics import YOLO
import cv2
import faiss
import numpy as np
from models import EmbeddedFeatureWrapper
from torch.utils.data import DataLoader
from torchvision import transforms
from data import CustomDataset
from torch.utils.data.dataloader import default_collate
from extract_features import extract_feature
from mobileone import mobileone, reparameterize_model
import torch
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image

def create_query_embedding(image_path, model, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Evaluate the model to get the embedding
    model.eval()
    model = model.to(device)
    model = reparameterize_model(model)
    with torch.no_grad():
        query_embedding = model(image_tensor)

    return query_embedding.cpu().numpy()


def retrieve_knn_faiss_euclidean(query_embeddings, db_embeddings, k):
    """
        Retrieve k nearest neighbors based on Euclidean distance using CPU.

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    index = faiss.IndexFlatL2(db_embeddings.shape[1])  # Euclidean distance index
    index.add(db_embeddings)
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return dists, retrieved_result_indices

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
    state_dict = torch.load("./finetuned_models/s2_56_epoch_45.pth", map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CustomDataset("D:/Coding/ygo_fsl/dataset/", transform=trans, train=True)
    if not os.path.exists("embeddings.npy"):

        dataloader = DataLoader(dataset, batch_size=8, num_workers=4,pin_memory=True, collate_fn=default_collate)
        embeddings, labels = extract_feature(loader=dataloader, model=model, device=device)
        # Save embeddings to a file
        np.save("embeddings.npy", embeddings)

    embeddings = np.load("embeddings.npy")
    binary_db_embeddings = np.require(embeddings > 0, dtype='float32')


    # Example usage
    

    ## Load YOLOv8 model
    # yolo_model = YOLO('./finetuned_models/yolo_ygo.pt')

    ## Load and prepare image
    # img_path = './real_life_images/ygo_field2.jpg'
    # img = cv2.imread(img_path)

    ## Run inference
    # results = yolo_model(img)

    ## Extract bounding box coordinates and confidence scores
    # detected_objects = []
    # for i, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):
    #     if conf >= 0.5:  # Filter by confidence (greater than 50%)
    #         x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
    #         # Crop the detected object from the image
    #         cropped_img = img[y1:y2, x1:x2]
    #         detected_objects.append(cropped_img)

    # # Create a batch of detected objects
    # batch_size = 8
    # detected_objects_batches = [detected_objects[i:i+batch_size] for i in range(0, len(detected_objects), batch_size)]

    # # Extract embeddings for each batch of detected objects
    # all_embeddings = []
    # for detected_objects_batch in detected_objects_batches:
    #     images = [Image.fromarray(obj) for obj in detected_objects_batch]
    #     image_tensors = [trans(img).unsqueeze(0).to(device) for img in images]
    #     image_tensors = torch.cat(image_tensors, dim=0)
    #     with torch.no_grad():
    #         embeddings_batch = model(image_tensors)
    #     all_embeddings.extend(embeddings_batch.cpu().numpy())

    # # # Convert embeddings to binary format
    # # # binary_query_embeddings = np.require(all_embeddings > 0, dtype='float32')
    # # binary_query_embeddings = np.stack([np.require(embedding > 0, dtype='float32') for embedding in all_embeddings], axis=0)

    # # # Retrieve k nearest neighbors for each detected object
    # # dists, retrieved_result_indices = retrieve_knn_faiss_euclidean(binary_query_embeddings, binary_db_embeddings, 1)

    # # # Convert retrieved indices to image names or image classes
    # # image_names = []
    # # image_classes = []
    # # for i in range(len(retrieved_result_indices)):
    # #     image_name = dataset.image_paths[retrieved_result_indices[i][0]]
    # #     image_names.append(image_name)
    # #     image_class = dataset.class_labels[retrieved_result_indices[i][0]]
    # #     image_classes.append(image_class)

    # # Display each detected object with its retrieved image
    # for i, obj in enumerate(detected_objects):  # zip(detected_objects, image_names, image_classes)):
    #     # window_name = f"Detected Object {i + 1}"
    #     # cv2.imshow(window_name, obj)
    #     cv2.imwrite(f"detected_object_{i + 1}.jpg", obj)
    #     # print(f"Detected Object {i + 1}: {name} with class {class_}")


    # # Press any key to close all windows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
