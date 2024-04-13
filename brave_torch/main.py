import torch
from torch import nn, Tensor
from zeta import FeedForward
from zeta.structs import ViTransformerWrapper, Encoder
from zeta import Attention


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
                    ),
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
            Tensor: The concatenated vision embeddings of shape (batch_size, num_of_vits * encoder_dim, s).

        """
        # IMG Shape
        b, c, h, w = x.shape

        # Vision Embedding list
        vision_embeddings = []

        for vit in self.vits:
            out = vit(x, return_embeddings=True)
            normed = self.norm(out)
            projected = self.proj(normed)
            vision_embeddings.append(projected)

        # Concat all of the tensors along S dimension
        # vision_embeddings = torch.concat(vision_embeddings, dim=1)
        # vision_embeddings.unsqueeze(1)
        vision_embeddings = torch.stack(vision_embeddings, dim=1)

        return vision_embeddings


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

        # Attention
        self.self_attn = Attention(
            dim,
            heads,
            causal=True,
            dropout=dropout,
            qk_norm=True,
            kv_heads=4,
        )

        # FeedForward
        self.ffn = FeedForward(
            dim,
            dim,
            mult,
            swish=True,
            post_act_ln=True,
            dropout=dropout,
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
        # attn_out = self.attn(x, self.queries, x)
        attn_out, _ = self.self_attn(x, self.queries)
        print(attn_out.shape)

        # Cross Attention on the output of the MultiQueryAttention
        cross_attn_out, _ = self.self_attn(attn_out, img)
        print(cross_attn_out.shape)

        # Feedforward
        feeded = self.ffn(cross_attn_out)
        print(feeded.shape)

        # Feeded
        # texted_ffn = self.ffn(x)

        # Concatenate
        # return torch.cat((feeded, texted_ffn), dim=1)
        return feeded


class BraveMultiModalFusion(nn.Module):
    """
    A module for performing multi-modal fusion using Brave ViT Swarm and MEQ Former.

    Args:
        dim (int): The dimension of the input.
        mult (int): The multiplier for the dimension of the input.
        depth (int, optional): The depth of the MEQ Former. Defaults to 1.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        heads (int, optional): The number of attention heads. Defaults to 8.
        image_size (int, optional): The size of the input image. Defaults to 256.
        patch_size (int, optional): The size of each patch in the image. Defaults to 32.
        encoder_dim (int, optional): The dimension of the encoder in the Brave ViT Swarm. Defaults to 512.
        encoder_depth (int, optional): The depth of the encoder in the Brave ViT Swarm. Defaults to 6.
        encoder_heads (int, optional): The number of attention heads in the encoder of the Brave ViT Swarm. Defaults to 8.
        num_of_vits (int, optional): The number of ViTs in the Brave ViT Swarm. Defaults to 4.
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        depth: int = 1,
        dropout: float = 0.1,
        heads: int = 8,
        image_size: int = 256,
        patch_size: int = 32,
        encoder_dim: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        num_of_vits: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.depth = depth
        self.dropout = dropout
        self.heads = heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.num_of_vits = num_of_vits

        # Vision
        self.vision = SwarmOfViTs(
            image_size=image_size,
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            num_of_vits=num_of_vits,
        )

        # MEQ Former
        self.meq = MeQFormer(
            dim,
            mult,
            depth,
            dropout,
            heads,
        )

    def forward(self, x: Tensor, img: Tensor) -> Tensor:
        # Vision -- apply the swarms of vision transformers
        vision_out = self.vision(img)[0]
        print(f"Vision out shape: {vision_out.shape}")

        # Text -- apply the MEQ Former
        text_out = self.meq(x, vision_out)
        print(f"Text out shape: {text_out.shape}")
        t_b, t_s, t_d = text_out.shape

        text_out = nn.Linear(t_d, self.dim)(text_out)

        return text_out
