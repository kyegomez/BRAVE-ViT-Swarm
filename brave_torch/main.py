import torch 
from torch import nn, Tensor
from zeta import FeedForward
from zeta.nn import MultiQueryAttention
from zeta.nn.attention.cross_attention import CrossAttention
from zeta.structs import ViTransformerWrapper, Encoder


class SwarmOfViTs(nn.Module):
    """
    A module that represents a swarm of Vision Transformers (ViTs).

    Args:
        image_size (int): The size of the input image.
        patch_size (int): The size of each patch in the image.
        encoder_dim (int): The dimension of the ViT encoder.
        encoder_depth (int): The depth of the ViT encoder.
        encoder_heads (int): The number of attention heads in the ViT encoder.
        num_of_vits (int): The number of ViTs in the swarm.

    Attributes:
        image_size (int): The size of the input image.
        patch_size (int): The size of each patch in the image.
        encoder_dim (int): The dimension of the ViT encoder.
        encoder_depth (int): The depth of the ViT encoder.
        encoder_heads (int): The number of attention heads in the ViT encoder.
        num_of_vits (int): The number of ViTs in the swarm.
        vits (nn.ModuleList): A list of ViTransformerWrapper instances.
        norm (nn.LayerNorm): A layer normalization module.
        proj (nn.Linear): A linear projection module.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        encoder_dim: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 8, 
        num_of_vits: int = 4,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.num_of_vits = num_of_vits
        
        # Create a list of ViTransformerWrapper instances
        self.vits = nn.ModuleList(
            [
                ViTransformerWrapper(
                    image_size=image_size,
                    patch_size=patch_size,
                    attn_layers=Encoder(
                        dim=encoder_dim,
                        depth=encoder_depth,
                        heads=encoder_heads,
                    )
                )
                for _ in range(num_of_vits)
            ]
        )
        
        
        # Norm
        self.norm = nn.LayerNorm(encoder_dim)
        
        # Projection
        self.proj = nn.Linear(encoder_dim, encoder_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SwarmOfViTs module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: The concatenated vision embeddings of shape (batch_size, num_of_vits * encoder_dim).

        """
        # IMG Shape
        b, c, h, w = x.shape
        
        # Norm and projection
        # x = self.norm(x)
        
        # Vision Embedding list
        vision_embeddings = []
        
        for vit in self.vits:
            out = vit(x, return_embeddings=True)
            b, s, d = out.shape
            normed = nn.LayerNorm(d)(out)
            projected = nn.Linear(d, d)(normed)
            vision_embeddings.append(projected)
        
        # Concat all of the tensors along S d
        vision_embeddings = torch.cat((vision_embeddings), dim=1)
            
        return vision_embeddings
            
            
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



class MeQFormer(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int,
        depth: int = 1,
        dropout: float = 0.1,
        heads: int = 8,
    ):
        """
        Multi-Query Transformer module that combines MultiQueryAttention, CrossAttention, and FeedForward layers.
        
        Args:
            dim (int): Dimension of the input and output tensors.
            mult (int): Multiplier for the dimension of the intermediate hidden layer in the FeedForward layer.
            depth (int, optional): Number of layers in the Multi-Query Transformer. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            heads (int, optional): Number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.depth = depth
        self.dropout = dropout
        
        # Initialize Attn
        self.attn = MultiQueryAttention(
            dim,
            heads,
        )
        
        # FeedForward
        self.ffn = FeedForward(
            dim,
            dim,
            mult,
            swish=True,
            post_act_ln=True,
            dropout=dropout
        )
        
        # CrossAttention
        self.cross_attn = CrossAttention(
            dim,
            dropout=dropout,
            heads=heads,
        )
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, heads, dim))
        
        # Learnable keys
        self.keys = nn.Parameter(torch.randn(1, heads, dim))
        
    def forward(self, x: Tensor, img: Tensor) -> Tensor:
        """
        Forward pass of the Multi-Query Transformer.
        
        Args:
            x (Tensor): Input tensor.
            img (Tensor): Image tensor.
        
        Returns:
            Tensor: Concatenated output tensor from the FeedForward and MultiQueryAttention layers.
        """
        # Attn
        attn_out = self.attn(x, self.queries, x)
        print(attn_out.shape)
        
        # Cross Attention on the output of the MultiQueryAttention
        cross_attn_out = self.cross_attn(attn_out,  img)
        
        # Feedforward
        feeded = self.ffn(cross_attn_out)
        
        # Feeded
        texted_ffn = self.ffn(x)
        
        # Concatenate
        return torch.cat((feeded, texted_ffn), dim=1)
    
    
