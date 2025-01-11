# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import os

import nibabel as nib
import numpy as np
import torch
from e3nn import o3
from e3nn.nn import SO3Activation


def s2_near_identity_grid(max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3) -> torch.Tensor:
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    return torch.stack((a.flatten(), b.flatten()))


def so3_near_identity_grid(
    max_beta: float = math.pi / 8, max_gamma: float = 2 * math.pi, n_alpha: int = 8, n_beta: int = 3, n_gamma=None
) -> torch.Tensor:
    if n_gamma is None:
        n_gamma = n_alpha
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha)[:-1]
    gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    a, b, c = torch.meshgrid(alpha, beta, gamma, indexing="ij")
    return torch.stack((a.flatten(), b.flatten(), c.flatten()))


def s2_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def flat_wigner(lmax: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [(2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1
    )


def save_features_as_nii(features, output_dir="nii_features"):
    """
    Save the extracted features as reshaped 2D .nii.gz files.

    Args:
        features: Torch tensor of shape [batch, features, irreps].
        output_dir: Directory to save the .nii.gz files.
    """
    os.makedirs(output_dir, exist_ok=True)

    features_np = features.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy
    print(np.shape(features_np))

    # Normalize features to [0, 1] with a small epsilon to avoid division by zero
    min_val = features_np.min(axis=1, keepdims=True)
    max_val = features_np.max(axis=1, keepdims=True)
    epsilon = 1e-8  # Small epsilon to prevent division by zero

    # Ensure the denominator doesn't become zero
    features_np = (features_np - min_val) / (max_val - min_val + epsilon)

    num_features, total_elements = features_np.shape  # [features, irreps]

    # Calculate the square dimension
    square_dim = int(math.sqrt(total_elements))
    if square_dim**2 != total_elements:
        raise ValueError(f"Feature size {total_elements} cannot be reshaped to a square grid.")

    reshaped_features = features_np.reshape(num_features, square_dim, square_dim)

    for i, feature_map in enumerate(reshaped_features):
        # Create a Nifti1Image for the feature map
        nii_image = nib.Nifti1Image(feature_map, affine=np.eye(4))
        # Save the .nii.gz file
        output_path = os.path.join(output_dir, f"feature_map_{i}.nii.gz")
        nib.save(nii_image, output_path)
        print(f"Saved feature map {i} to {output_path}")


class S2Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]
        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
        )  # [n_s2_pts, psi]
        self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]
        self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, psi]
        self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class S2ConvNetModified(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 96
        lmax1 = 10

        b_l1 = 10
        lmax2 = 5

        b_l2 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.from_s2 = o3.FromS2Grid((b_in, b_in), lmax1)

        self.conv1 = S2Convolution(1, f1, lmax1, kernel_grid=grid_s2)

        self.act1 = SO3Activation(lmax1, lmax2, torch.relu, b_l1)

        self.conv2 = SO3Convolution(f1, f2, lmax2, kernel_grid=grid_so3)

        self.act2 = SO3Activation(lmax2, 0, torch.relu, b_l2)

        self.w_out = torch.nn.Parameter(torch.randn(f2, f_output))

    def forward(self, x):
        x = x.transpose(-1, -2)  # [batch, features, alpha, beta] -> [batch, features, beta, alpha]

        x = self.from_s2(x)  # [batch, features, beta, alpha] -> [batch, features, irreps]
        x = self.conv1(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.act1(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.conv2(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.act2(x)  # [batch, features, scalar]

        # x = x.flatten(1) @ self.w_out / self.w_out.shape[0]

        return x


def load_nii_data(file_path, index, dimension):
    """
    Load a 3D .nii.gz file, extract a specific slice, and prepare it for the network.
    """
    nii_data = nib.load(file_path)
    volume = nii_data.get_fdata()

    # Select the slice along the specified dimension
    if dimension == 0:  # Axial
        slice_2d = volume[index, :, :]
    elif dimension == 1:  # Coronal
        slice_2d = volume[:, index, :]
    elif dimension == 2:  # Sagittal
        slice_2d = volume[:, :, index]
    else:
        raise ValueError("Dimension must be 0 (Axial), 1 (Coronal), or 2 (Sagittal).")

    # Normalize the slice and add necessary dimensions
    slice_2d = (slice_2d - np.mean(slice_2d)) / np.std(slice_2d)
    slice_2d = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    return slice_2d


def main():
    """
    Equivariant feature extractor that loads in a 3D nii.gz image, extracts a single slice and
    pushes it through the equivariant network. The extracted features are printed to terminal.
    """
    nii_file_path = "testing_data/source_0_0.nii.gz"  # Path to the 3D .nii.gz file
    slice_index = 64  # Index of the slice to extract
    dimension = 0  # 0 = Axial, 1 = Coronal, 2 = Sagittal

    # Load and process the 2D slice from the 3D volume
    input_slice = load_nii_data(nii_file_path, slice_index, dimension)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S2ConvNetModified().to(device)

    input_slice = input_slice.to(device)  # Move to the appropriate device
    with torch.no_grad():
        features = model(input_slice)

    print("Extracted features:", features)  # print out extracted features from the equivariant filter

    # Save features as .nii.gz files
    # save_features_as_nii(features, output_dir="nii_features")


if __name__ == "__main__":
    main()
