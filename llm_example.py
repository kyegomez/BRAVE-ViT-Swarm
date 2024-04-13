import torch 
from brave_torch.llm import LLM

x = torch.randint(0, 256, (1, 1000))

img = torch.randn(1, 3, 256, 256)

model = LLM(
    dim=512,
    depth=1,
    num_tokens=256,
    dim_head=64,
    heads=8,
    ff_mult=4,
    image_size=256,
    patch_size=32,
    encoder_dim=512,
    encoder_depth=6,
    encoder_heads=8,
    num_of_vits=4,
)

out = model(x, img)
print(out.shape)