# import torch
# import torch.nn as nn

# class HashingLayer(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#     ) -> None:
#         super().__init__()

#         # Random assignment: each input -> exactly one output
#         mask = torch.randint(
#             low=0,
#             high=out_features,
#             size=(in_features,)
#         )

#         # Build sparse indices
#         indices = torch.stack([
#             mask,                          # row indices (outputs)
#             torch.arange(in_features)      # col indices (inputs)
#         ], dim=0)

#         values = torch.ones(in_features)

#         sparse_weight = torch.sparse_coo_tensor(
#             indices,
#             values,
#             size=(out_features, in_features)
#         ).coalesce()

#         # Remove dense parameter and replace with sparse buffer
#         self.register_buffer("weight", sparse_weight)

#     def forward(self, x):
#         # Ensure x is 2D: (batch, in_features)
#         if x.dim() == 1:
#             x = x.unsqueeze(0)

#         out = torch.sparse.mm(self.weight, x.T).T

#         # # If original input was 1D, return 1D
#         # if out.shape[0] == 1:
#         #     out = out.squeeze(0)

#         return out

import torch
import torch.nn as nn

class HashingLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, seed: int = None):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        # Random assignment: each input -> exactly one output
        mask = torch.randint(
            low=0,
            high=out_features,
            size=(in_features,)
        )

        # Random ±1 signs
        signs = torch.randint(0, 2, (in_features,)) * 2 - 1  # 0->-1, 1->+1
        values = signs.float()

        # Build sparse indices
        indices = torch.stack([
            mask,                          # row indices (outputs)
            torch.arange(in_features)      # col indices (inputs)
        ], dim=0)

        sparse_weight = torch.sparse_coo_tensor(
            indices,
            values,
            size=(out_features, in_features)
        ).coalesce()

        # Register sparse weight as buffer
        self.register_buffer("weight", sparse_weight)

    def forward(self, x):
        # Ensure x is 2D: (batch, in_features)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = torch.sparse.mm(self.weight, x.T).T
        return out
