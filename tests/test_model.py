from models import EmbeddedFeatureWrapper, GeM
from mobileone import reparameterize_model, mobileone
import torch
import os


def test_model_forward_pass():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
    model.feature.gap = GeM()

    # Create a dummy input tensor with the expected shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 56, 56)  # Example input shape for MobileOne

    # Perform a forward pass
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(dummy_input)

    assert output.shape[0] == 1, "Output batch size is incorrect"
    assert output.shape[1] == 2048, "Output feature dimension is incorrect"

def test_model_reparameterization():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = mobileone(variant="s2")
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input tensor with the expected shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 56, 56)  # Example input shape for MobileOne

    # Perform a forward pass before reparameterization
    with torch.no_grad():
        output_before = model(dummy_input)

    # Reparameterize the model
    reparameterized_model = reparameterize_model(model)
    reparameterized_model.eval()  # Set the reparameterized model to evaluation mode

    # Perform a forward pass after reparameterization
    with torch.no_grad():
        output_after = reparameterized_model(dummy_input)

    # Check if the outputs are approximately equal
    assert torch.allclose(output_before, output_after, atol=1e-5), "Outputs before and after reparameterization are not close enough"