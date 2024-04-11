[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BRAVE or Swarms of Vision Transformers
Implementation of the paper: "BRAVE : Broadening the visual encoding of vision-language models". BRAVE achieves state-of-the-art performance on a broad range of captioning and VQA benchmarks and significantly reduces the aforementioned issues of VLMs, while requiring a smaller number of trainable parameters than existing methods and having a more compressed representation

## install
`pip3 install brave-torch`


## usage

### 
```python
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
```

# Citations

## Todo
- [ ] Citation link
- [ ] Citation Bibtex
- [ ] Diagram photo
- [ ] Implement Andromeda Base LLM architecture
- [ ] Provide multi-modal tokenizer
- [ ] Train and release the model 