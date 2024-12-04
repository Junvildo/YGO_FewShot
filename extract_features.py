import numpy as np
import torch
import time
from mobileone import reparameterize_model


def extract_feature(model, loader, device):
    """
    Extract embeddings from given `model` for given `loader` dataset on `device`.
    """
    model.eval()
    model_eval = reparameterize_model(model)
    model_eval = model_eval.to(device=device)
    
    all_embeddings = []
    all_labels = []
    log_every_n_step = 10
    start = time.time()
    with torch.no_grad():
        for i, (im, instance_label, _) in enumerate(loader):
            im = im.to(device=device)
            embedding = model_eval(im)

            all_embeddings.append(embedding.cpu().numpy())
            all_labels.extend(instance_label.tolist())

            if (i + 1) % log_every_n_step == 0:
                print('Process Iteration {} / {}'.format(i, len(loader)))

    all_embeddings = np.vstack(all_embeddings)
    end = time.time()
    print("Generated {} embedding matrix".format(all_embeddings.shape))
    print("Generate {} labels".format(len(all_labels)))
    print("Time cost: {} seconds".format(end - start))

    model.train()
    return all_embeddings, all_labels