[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BRAVE or Swarms of Vision Transformers
Implementation of the paper: "BRAVE : Broadening the visual encoding of vision-language models". BRAVE achieves state-of-the-art performance on a broad range of captioning and VQA benchmarks and significantly reduces the aforementioned issues of VLMs, while requiring a smaller number of trainable parameters than existing methods and having a more compressed representation.

## install
`pip3 install brave-torch`


## usage
`pip3 install brave-torch`

## `LLM`
- A fully ready to train LLM with the Swarm of Vits + MEQFormer

```python
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
```




### `BraveMultiModalFusion`
- The Swarm of ViTs coupled with the meqformer 

```python
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

```

# Citations

## Todo
- [ ] Citation link
- [ ] Citation Bibtex
- [ ] Diagram photo
- [ ] Implement Andromeda Base LLM architecture
- [ ] Provide multi-modal tokenizer
- [ ] Train and release the model 