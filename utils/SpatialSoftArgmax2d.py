# PyTorch geometry (v0.7.0) [216aa9d]: https://github.com/kornia/kornia
# Also see, original implementation: https://github.com/gorosgobe/dsae-torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def create_meshgrid(height: int, width: int, normalized_coordinates: bool = True, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
	Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        torch.Tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
	
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
	
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2

def spatial_softmax2d(input: torch.Tensor, temperature: torch.Tensor = torch.tensor(1.0)) -> torch.Tensor:
    """
    Apply the Softmax function over features in each image channel.

    Note that this function behaves differently to :py:class:`torch.nn.Softmax2d`, which
    instead applies Softmax over features at each spatial location.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        temperature: factor to apply to input, adjusting the "smoothness" of the output distribution.

    Returns:
       a 2D probability distribution per image channel with shape :math:`(B, N, H, W)`.

    Examples:
        >>> heatmaps = torch.Tensor([[[
        ... [0., 0., 0.],
        ... [0., 0., 0.],
        ... [0., 1., 2.]]]])
        >>> spatial_softmax2d(heatmaps)
        tensor([[[[0.0585, 0.0585, 0.0585],
                  [0.0585, 0.0585, 0.0585],
                  [0.0585, 0.1589, 0.4319]]]])
    """
    batch_size, channels, height, width = input.shape
    temperature = temperature.to(device=input.device, dtype=input.dtype)
    x = input.view(batch_size, channels, -1)

    x_soft = F.softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, height, width)

def spatial_expectation2d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    """
    Compute the expectation of coordinate values using spatial probabilities.

    The input heatmap is assumed to represent a valid spatial probability distribution,
    which can be achieved using :func:`~kornia.geometry.subpixel.spatial_softmax2d`.

    Args:
        input: the input tensor representing dense spatial probabilities with shape :math:`(B, N, H, W)`.
        normalized_coordinates: whether to return the coordinates normalized in the range
          of :math:`[-1, 1]`. Otherwise, it will return the coordinates in the range of the input shape.

    Returns:
       expected value of the 2D coordinates with shape :math:`(B, N, 2)`. Output order of the coordinates is (x, y).

    Examples:
        >>> heatmaps = torch.Tensor([[[
        ... [0., 0., 0.],
        ... [0., 0., 0.],
        ... [0., 1., 0.]]]])
        >>> spatial_expectation2d(heatmaps, False)
        tensor([[[1., 2.]]])
    """
    batch_size, channels, height, width = input.shape

    # Create coordinates grid.
    grid = create_meshgrid(height, width, normalized_coordinates, input.device)
    grid = grid.to(input.dtype)

    pos_x = grid[..., 0].reshape(-1)
    pos_y = grid[..., 1].reshape(-1)

    input_flat = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output = torch.cat([expected_x, expected_y], dim=-1)

    return output.view(batch_size, channels, 2)  # BxNx2

def spatial_soft_argmax2d(
    input: torch.Tensor, temperature: torch.Tensor = torch.tensor(1.0), normalized_coordinates: bool = True
) -> torch.Tensor:
    """
    Compute the Spatial Soft-Argmax 2D of a given input heatmap.

    Args:
        input: the given heatmap with shape :math:`(B, N, H, W)`.
        temperature: factor to apply to input.
        normalized_coordinates: whether to return the coordinates normalized in the range of :math:`[-1, 1]`.
            Otherwise, it will return the coordinates in the range of the input shape.

    Returns:
        the index of the maximum 2d coordinates of the give map :math:`(B, N, 2)`.
        The output order is x-coord and y-coord.

    Examples:
        >>> input = torch.Tensor([[[
        ... [0., 0., 0.],
        ... [0., 10., 0.],
        ... [0., 0., 0.]]]])
        >>> spatial_soft_argmax2d(input, normalized_coordinates=False)
        tensor([[[1.0000, 1.0000]]])
    """
    input_soft: torch.Tensor = spatial_softmax2d(input, temperature)
    output: torch.Tensor = spatial_expectation2d(input_soft, normalized_coordinates)
    return output

class SpatialSoftArgmax2d(nn.Module):
    """
    Compute the Spatial Soft-Argmax 2D of a given heatmap.

    See :func:`~kornia.geometry.subpix.spatial_soft_argmax2d` for details.
    """

    def __init__(self, temperature: torch.Tensor = torch.tensor(1.0), normalized_coordinates: bool = True) -> None:
        super().__init__()
        self.temperature: torch.Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"temperature={self.temperature}, "
            f"normalized_coordinates={self.normalized_coordinates})"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return spatial_soft_argmax2d(input, self.temperature, self.normalized_coordinates)
