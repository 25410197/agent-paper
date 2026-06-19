---
name: feedback-vit-assumptions
description: Assumptions for ViT-based papers that don't specify embed_dim/num_layers/num_heads — reverse-engineer from total params
metadata:
  type: feedback
---

When a ViT-based paper (like MaizeFormerX) specifies total params (e.g., 9.6M) but does NOT state embed_dim, num_layers, or num_heads, use reverse engineering:

**Rule:** Estimate D, L, H by solving approximate param budget equations:
- PatchEmbed per scale: ~3 * patch_size^2 * D + D (conv weights + bias)
- CSAF: ~3 * (D^2 / H) * H * 3 = ~3 * D^2 cross-attention matrices
- TransformerLayer: ~4 * D^2 (MHSA) + 2 * D * (4*D) (FFN) ≈ 12 * D^2
- Positional embeddings: N * D per scale

**Default starting point for ~10M param ViT:** D=256, L=6, H=8 — this combination typically yields 8-12M params for custom lightweight ViTs.

**Why:** Papers often specify "L layers" and "D token dimension" without values, claiming "Algorithm 1" which is not always in parseable text form.

**How to apply:** Always document the assumption as HIGH risk in ReproduceSummary.md. Flag link to official code repo if available. Check `torchinfo.summary()` output vs paper's reported param count after running.
