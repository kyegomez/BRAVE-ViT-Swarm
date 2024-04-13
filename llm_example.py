import torch  # Importing the torch library
from brave_torch.llm import LLM  # Importing the LLM class from brave_torch.llm module

x = torch.randint(0, 256, (1, 1000))  # Generating a random tensor 'x' with values between 0 and 256

img = torch.randn(1, 3, 256, 256)  # Generating a random image tensor 'img' with shape (1, 3, 256, 256)

model = LLM(
    dim=512,  # Dimension of the model
    depth=1,  # Depth of the model
    num_tokens=256,  # Number of tokens
    dim_head=64,  # Dimension of the attention head
    heads=8,  # Number of attention heads
    ff_mult=4,  # Multiplier for the feed-forward network dimension
    image_size=256,  # Size of the input image
    patch_size=32,  # Size of the image patch
    encoder_dim=512,  # Dimension of the encoder
    encoder_depth=6,  # Depth of the encoder
    encoder_heads=8,  # Number of attention heads in the encoder
    num_of_vits=4,  # Number of ViTs (Vision Transformers)
)

out = model(x, img)  # Forward pass through the model
print(out.shape)  # Printing the shape of the output tensor