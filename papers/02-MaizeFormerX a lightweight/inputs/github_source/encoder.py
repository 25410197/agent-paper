"""
Transformer encoder blocks for MaizeFormerX.

Architecture: Pre-norm transformer encoder.
Each block: x + DropPath(LayerScale(Attention(Norm(x)))) + DropPath(LayerScale(MLP(Norm(x))))

Key confirmed parameters (from model.py defaults):
  dim         = 192  (fusion_dim)
  depth       = 6
  num_heads   = 6
  mlp_ratio   = 4.0
  drop_path   = 0.1  (linearly scheduled across depth)
  attn_dropout = 0.1
  proj_dropout = 0.1
  qkv_bias    = True
  layer_scale = False

NOTE: Full source not fetched (imports from src.models.common.*).
      Parameters above are confirmed from model.py constructor defaults.
      Architecture description sourced from GitHub WebFetch summary.
"""

# TransformerEncoderBlock (single layer):
#   norm1 -> MultiHeadSelfAttention -> DropPath -> residual
#   norm2 -> MLP(dim, mlp_ratio*dim, GELU) -> DropPath -> residual
#   Optional: LayerScale on attention and MLP outputs
#
# TransformerEncoder (stack of blocks):
#   drop_path rates linearly interpolated from 0 -> drop_path across depth
#   Final LayerNorm applied after all blocks
