import numpy as np

import torch

from mobileone import reparameterize_model


def extract_feature(model, loader, device):
    """
    Extract embeddings from given `model` for given `loader` dataset on `device`.
    """
    model.eval()

    all_embeddings = []
    all_labels = []
    log_every_n_step = 10

    with torch.no_grad():
        for i, (im, instance_label) in enumerate(loader):
            im = im.to(device=device)
            embedding = model(im)

            all_embeddings.append(embedding.cpu().numpy())
            all_labels.extend(instance_label.tolist())

            if (i + 1) % log_every_n_step == 0:
                print('Process Iteration {} / {}'.format(i, len(loader)))

    all_embeddings = np.vstack(all_embeddings)

    print("Generated {} embedding matrix".format(all_embeddings.shape))
    print("Generate {} labels".format(len(all_labels)))

    model.train()
    return all_embeddings, all_labels