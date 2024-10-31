from mobileone import mobileone
from torchsummary import summary
from torch import nn

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


baseline = mobileone(variant="s2")
model = EmbeddedFeatureWrapper(feature=baseline, input_dim=2048, output_dim=2048)

model = mobileone(variant="s2")
summary(model, (3, 7, 7))