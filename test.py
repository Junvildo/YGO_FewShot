from extract_features import extract_feature
import os
import torch
import faiss
from torchvision import transforms
from models import EmbeddedFeatureWrapper, GeM
from mobileone import mobileone, reparameterize_model
from data import CustomDataset
from PIL import Image
import numpy as np
import random
import antialiased_cnns
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_similar_images(query_embedding, base_dataset, index, num_results=4):
    """
    Given a query embedding, find the most similar images in the dataset.
    """
    distances1, indices1 = index.search(query_embedding, num_results)
    image_name = [base_dataset.class_labels_list[i] for i in indices1[0]]
    print(image_name)

    image_names = [base_dataset.label_to_classname[base_dataset.class_labels_list[i]] for i in indices1[0]]
    image_path = ['https://images.ygoprodeck.com/images/cards_cropped/{}.jpg'.format(image_name) for image_name in image_names]
    print(set(image_path))


# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)

model.feature.gap = GeM()
# model.feature.gap = torch.nn.Sequential(antialiased_cnns.BlurPool(2048, stride=2, filt_size=1), GeM())

state_dict = torch.load("finetuned_models/s2_56_color_mixup_gem_aug_best.pth", map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# print(state_dict.keys())
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
model_eval = reparameterize_model(model)
model_eval.eval()

mean, std = [0.49362021684646606, 0.4601792097091675, 0.4618436098098755], [0.27437326312065125, 0.2629182040691376, 0.270280659198761]
trans = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

data_root = "dataset"
base_dataset = CustomDataset(root=data_root, train=True, transform=trans)

# label2name = json.load(open("label2name.json", "r"))

img_path = "real_life_images/dark_infant.png"
img_path = "real_life_images/detected_object_2.jpg"
# img_path = "dataset/test/2511/2511_original.jpg"
img = Image.open(img_path).convert('RGB')

with torch.no_grad():
    query_embedding = model_eval(torch.unsqueeze(trans(img), 0).to(device))

index_path_nogem = ["precompute_embs/cutmixup/avg_pool/full_class_embeddings_56.faiss"
                    ,"precompute_embs/cutmixup/avg_pool/full_class_embeddings_f_56.faiss"
                    ,"precompute_embs/cutmixup/avg_pool/org_class_embeddings_56.faiss"
                    ,"precompute_embs/cutmixup/avg_pool/org_class_embeddings_56_f.faiss"]

index_path_gem = ["precompute_embs/cutmixup/better_aug/full_class_embeddings_56.faiss"
                  ,"precompute_embs/cutmixup/better_aug/full_class_embeddings_f_56.faiss"
                  ,"precompute_embs/cutmixup/better_aug/org_class_embeddings_56.faiss"
                  ,"precompute_embs/cutmixup/better_aug/org_class_embeddings_f_56.faiss"]

index_path_gem_progressive = ["precompute_embs/cutmixup/better_aug/progressive/full_class_embeddings_224.faiss",
                            "precompute_embs/cutmixup/better_aug/progressive/full_class_embeddings_f_224.faiss",
                            "precompute_embs/cutmixup/better_aug/progressive/org_class_embeddings_224.faiss",
                            "precompute_embs/cutmixup/better_aug/progressive/org_class_embeddings_f_224.faiss"]

index_path_gem_blur_progressive = ["precompute_embs/cutmixup/better_aug/blur_pool/full_class_embeddings_224.faiss",
                                  "precompute_embs/cutmixup/better_aug/blur_pool/full_class_embeddings_f_224.faiss",
                                  "precompute_embs/cutmixup/better_aug/blur_pool/org_class_embeddings_224.faiss",
                                  "precompute_embs/cutmixup/better_aug/blur_pool/org_class_embeddings_f_224.faiss"]
# index = faiss.IndexFlatL2(2048)  # Euclidean distance index
# index = faiss.read_index(index_path_nogem[0])
# index_f = faiss.IndexFlatIP(2048)
# index_f = faiss.read_index(index_path_nogem[1])

index_gem = faiss.IndexFlatL2(2048)
index_gem = faiss.read_index(index_path_gem[0])
index_f_gem = faiss.IndexFlatIP(2048)
index_f_gem = faiss.read_index(index_path_gem[1])

# print("Avg Pool:")
# print("Binary:")
# binary_db_embeddings = np.require(query_embedding > 0, dtype='float32')
# get_similar_images(binary_db_embeddings, base_dataset, index, num_results=4)
# print("Float:")
# get_similar_images(query_embedding, base_dataset, index_f, num_results=4)

print("Gem Pool:")
print("Binary:")
binary_db_embeddings = np.require(query_embedding > 0, dtype='float32')
get_similar_images(binary_db_embeddings, base_dataset, index_gem, num_results=10)
print("Float:")
get_similar_images(query_embedding, base_dataset, index_f_gem, num_results=10)