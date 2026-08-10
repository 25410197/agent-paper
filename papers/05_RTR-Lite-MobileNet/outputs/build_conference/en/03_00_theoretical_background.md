## 2. Theoretical Background and Related Work

### 2.1. Lightweight CNNs and MobileNetV2

With an input feature map of size $(H \times W \times C_{in})$, kernel $(k \times k)$ and $C_{out}$ output channels, a standard convolution requires approximately

$$HWk^2C_{in}C_{out}$$

multiply-accumulate operations. Depthwise separable convolution decomposes this computation into a per-channel depthwise convolution and a $1\times1$ pointwise convolution that mixes information across channels. The cost is then approximately

$$HW\left(k^2C_{in}+C_{in}C_{out}\right),$$

giving a ratio relative to standard convolution of

$$\frac{1}{C_{out}}+\frac{1}{k^2}.$$

This is the basis for how the MobileNet family reduces computation on resource-constrained devices. MobileNetV2 organizes these convolutions inside the *inverted residual with linear bottleneck* block (Sandler et al., 2018). Each block first uses a pointwise convolution to expand the channel count, then a depthwise convolution to extract spatial features, and finally a linear projection back to a low-dimensional space. When the stride equals 1 and the input/output sizes are compatible, a shortcut connects the two thin bottleneck representations. The final projection uses no activation function to limit information loss in the low-dimensional space. ChiliEdgeNet inherits this local inductive bias and computational efficiency from the retained portion of MobileNetV2.

### 2.2. Image Tokens and Self-Attention

Let the CNN backbone produce a feature map $F\in\mathbb{R}^{C\times H\times W}$. After a channel projection, the feature map is flattened into a sequence

$$X\in\mathbb{R}^{N\times d},\qquad N=HW,$$

where each row is a token representing one spatial location. From $X$, the query, key, and value matrices are formed by

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

Scaled dot-product attention (Vaswani et al., 2017) is defined as

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}\right)V,$$

where $d_h$ is the dimensionality of each attention head. Unlike convolution, which aggregates only a local neighbourhood directly at each layer, self-attention lets every token interact directly with every other token. In exchange, the attention matrix has $O(N^2)$ complexity in the number of tokens. ChiliEdgeNet uses a $14\times14$ grid, i.e. $N=196$, to retain more spatial positions than a $7\times7$ grid while still bounding the cost of a small Transformer block.

The Transformer block in this work uses a pre-norm structure with two residual branches:

$$X' = X + \operatorname{GQA}(\operatorname{RMSNorm}(X)),$$

$$Y = X' + \operatorname{MLP}(\operatorname{RMSNorm}(X')).$$

The MLP consists of two linear layers, GELU, and dropout. The residual connection provides a direct path for representations and gradients; pre-norm normalizes the input of each transformation rather than normalizing after the addition.

### 2.3. Grouped-Query Attention

Standard multi-head attention creates separate keys and values for every query head. Grouped-query attention (GQA) sets the number of query heads $h_q$ larger than the number of key/value heads $h_{kv}$, then lets a group of $h_q/h_{kv}$ query heads share one key/value pair (Ainslie et al., 2023). GQA sits between multi-head attention, where $h_q=h_{kv}$, and multi-query attention, where every query head shares a single key/value head.

If the embedding has dimension $d$ and the projections use no bias, the query and output projections of GQA still require $2d^2$ parameters, while the key and value require

$$2d^2\frac{h_{kv}}{h_q}.$$

The total parameter count of the four attention projections is therefore

$$P_{GQA}=2d^2+2d^2\frac{h_{kv}}{h_q}.$$

With $d=128$, $h_q=4$, and $h_{kv}=2$, each key/value head serves two query heads and the module avoids **16,384** parameters compared with four-head multi-head attention. In this work's implementation, the key and value are projected with two heads and then repeated by group before computing attention. The benefit asserted here is therefore a reduction in projection parameters; we do not extrapolate this into a conclusion about inference-speed gains.

### 2.4. 2D-RoPE Positional Encoding and RMSNorm

Self-attention carries no inherent information about token order or coordinates. Rotary position embedding (RoPE) encodes position by rotating each pair of components of the query and key by a position-dependent angle (Su et al., 2024). For a two-dimensional pair $z=(z_1,z_2)$, the rotation at position $p$ has the form

$$R(p\theta)z=\begin{bmatrix}\cos(p\theta) & -\sin(p\theta)\\\sin(p\theta) & \cos(p\theta)\end{bmatrix}z.$$

The original RoPE formulation is presented for one-dimensional sequences (Su et al., 2024). For an image grid, this work uses **axial 2D-RoPE**: each query/key head is split into two halves; one half is rotated by the row coordinate ($y$) and the other by the column coordinate ($x$), following the RoPE-Axial extension for Vision Transformers of Heo et al. (2025). This injects 2D coordinates into the query and key without learning a positional-embedding table. With 196 tokens and embedding dimension 128, a learned positional table would require $196\times128=25{,}088$ parameters.

RMSNorm normalizes by the root-mean-square rather than subtracting the mean as LayerNorm does (Zhang and Sennrich, 2019). For a token $x\in\mathbb{R}^{d}$, the normalization used in the model is

$$\operatorname{RMSNorm}(x)_i=g_i\frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\varepsilon}},$$

where $g_i$ is a learned scale coefficient and $\varepsilon=10^{-6}$. The implementation has no shift bias. ChiliEdgeNet has three normalization points of dimension 128, so RMSNorm avoids $3\times128=384$ bias parameters relative to three equivalent LayerNorms. This is an exact structural difference; RMSNorm's effect on accuracy still must be confirmed by ablation rather than inferred directly from the formula.

### 2.5. Transfer Learning and Few-Shot Domain Adaptation

Transfer learning leverages knowledge from a source domain or task to support a target domain or task (Pan and Yang, 2010). This work involves two levels of transfer. First, the MobileNetV2 backbone uses ImageNet-pretrained weights before being fine-tuned on ChiliEdge-DS. Second, the cross-dataset experiment treats ChiliEdge-DS as the source domain and the Mendeley Data image set as the target domain, which differs in acquisition conditions, background, and symptom distribution.

Let the source training set be $\mathcal{D}_s$, the labelled support set of the target domain be $S_t$, and the target-domain query set be $Q_t$. The current adaptation procedure forms

$$\mathcal{D}_{adapt}=\mathcal{D}_s\cup S_t$$

and optimizes the average cross-entropy over this set:

$$\mathcal{L}_{CE}=-\frac{1}{|\mathcal{D}_{adapt}|}\sum_{(x,y)\in\mathcal{D}_{adapt}}\log p_{\theta}(y\mid x).$$

After adaptation, $Q_t$ is used only for evaluation. The minimal independence condition is

$$S_t\cap Q_t=\varnothing,$$

verified in this work by SHA-256 hashes of the image files. If a query image were present in the support set, the result would no longer reflect genuine generalization to unseen target images.

In episodic few-shot learning, an episode typically comprises an $N$-way $K$-shot problem with a controlled number of support examples per class; Prototypical Networks is one example of metric learning under this setup (Snell et al., 2017). The experiments in this work are not trained episodically, do not learn prototypes, and have an uneven number of support examples across the four classes. The precise term used in this paper is therefore **supervised domain adaptation with a few-shot support set**, not meta-learning or balanced $N$-way $K$-shot.
