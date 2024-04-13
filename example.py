import torch  # Importing the torch library for deep learning operations
from brave_torch.main import (
    BraveMultiModalFusion,
)  # Importing the BraveMultiModalFusion class from brave_torch.main module

x = torch.randn(
    1, 1000, 512
)  # Generating a random tensor of shape (1, 1000, 512) using torch.randn
img = torch.randn(
    1, 3, 256, 256
)  # Generating a random tensor of shape (1, 3, 256, 256) using torch.randn

model = BraveMultiModalFusion(
    dim=512,  # Dimension of the model
    mult=4,  # Multiplier for the dimension
    depth=1,  # Depth of the model
    dropout=0.1,  # Dropout rate
    heads=8,  # Number of attention heads
    image_size=256,  # Size of the input image
    patch_size=32,  # Size of the image patches
    encoder_dim=512,  # Dimension of the encoder
    encoder_depth=6,  # Depth of the encoder
    encoder_heads=8,  # Number of attention heads in the encoder
    num_of_vits=4,  # Number of ViTs (Vision Transformers)
)

out = model(
    x, img
)  # Forward pass through the model to get the output
print(out)  # Printing the output
