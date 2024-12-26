from torch import nn
import torch
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class EmbeddedFeatureWrapper(nn.Module):
    """
    Wraps a base model with embedding layer modifications.
    """
    def __init__(self,
                 feature,
                 input_dim,
                 output_dim):
        super(EmbeddedFeatureWrapper, self).__init__()

        self.feature = nn.Sequential(
            feature.stage0,
            feature.stage1,
            feature.stage2,
            feature.stage3,
            feature.stage4,
            feature.gap
        )
        self.standardize = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.remap = None
        if input_dim != output_dim:
            self.remap = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, images):
        x = self.feature(images)
        x = x.view(x.size(0), -1)
        x = self.standardize(x)

        if self.remap:
            x = self.remap(x)

        x = nn.functional.normalize(x, dim=1)

        return x

    def __str__(self):
        return "{}_{}".format(self.feature.name, str(self.embed))