
import torch
from brave_torch.main import SwarmOfViTs

# IMG Tensor
x = torch.randn(1, 3, 224, 224) 

# Model
model = SwarmOfViTs(
    image_size=224,
    patch_size=32,
    encoder_dim=512,
    encoder_depth=6,
    encoder_heads=8,
    num_of_vits=4
)

# Forward
out = model(x)
print(out)