import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    B, H, W, C = image.shape
    P = patch_size

    num_patches_h = H // P
    num_patches_w = W // P
    N = num_patches_h * num_patches_w

    patches = image.reshape(B, num_patches_h, P, num_patches_w, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5) # B, h, w, P, P, C
    patches = patches.reshape(B, N, P*P*C)

    if W_proj is None:
        W_proj = np.random.randn(P * P * C, embed_dim) * 0.02

    return patches @ W_proj