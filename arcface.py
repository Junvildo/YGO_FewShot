import math

import torch
from torch import nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=64, margin=0.5, easy_margin=False, device='cpu', variant=1):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).
        
        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.variant = variant
        self.device = device
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        if self.variant == 1:
            return self.forward1(embedding, ground_truth)
        elif self.variant == 2:
            return self.forward2(embedding, ground_truth)
        elif self.variant == 3:
            return self.forward3(embedding, ground_truth)

    def forward1(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is from https://github.com/ronghuaiyang/arcface-pytorch, which takes
        54.804054962005466 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size(), device=self.device)
        one_hot.scatter_(1, ground_truth.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cos_theta)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.scale

        loss = self.ce(output, ground_truth)
        return loss

    def forward2(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is from https://github.com/deepinsight/insightface, which takes
        66.45489303627983 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080 Ti.
        Please noted that, different with forward1&3, this implementation ignore the samples that
        caused \theta + m > \pi to happen if easy_margin is False. And if easy_margin is True,
        it will do nothing even if \theta + m > \pi.
        """
        embedding = F.normalize(embedding)
        w = F.normalize(self.weight)
        cos_theta = F.linear(embedding, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if self.easy_margin:
            mask = torch.ones_like(ground_truth)
        else:
            mask = torch.gather(cos_theta, 1, ground_truth.view(-1, 1)).view(-1)
            mask = torch.where(mask.acos_() + self.margin > math.pi, 0, 1)
        mask = torch.where(mask != 0)[0]
        m_hot = torch.zeros(mask.shape[0], cos_theta.shape[1], device=cos_theta.device)
        m_hot.scatter_(1, ground_truth[mask, None], self.margin)
        theta = cos_theta.acos()
        output = (theta + m_hot).cos()
        output.mul_(self.scale)
        loss = self.ce(output[mask], ground_truth[mask])
        return loss

    def forward3(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is modified from forward1, which takes
        52.49644996365532 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080 Ti.
        """
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        pos = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        sin_theta = torch.sqrt((1.0 - torch.pow(pos, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = pos * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, pos)
        else:
            phi = torch.where(pos > self.th, phi, pos - self.mm)
        # one_hot = torch.zeros(cos_theta.size(), device='cuda')
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        # output = cos_theta + one_hot
        output *= self.scale
        loss = self.ce(output, ground_truth)
        return loss
