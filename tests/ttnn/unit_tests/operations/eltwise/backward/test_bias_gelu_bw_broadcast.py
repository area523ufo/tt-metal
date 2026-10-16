# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "input_shape,bias_shape",
    [
        ((2, 4, 32, 64), (1, 1, 1, 64)),
        ((1, 2, 64, 32), (1, 1, 64, 32)),
        ((1, 1, 32, 32), (1, 1, 32, 32)),
    ],
)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_bias_gelu_bw_broadcast_and_independent_buffers(device, input_shape, bias_shape, approximate):
    torch.manual_seed(0)
    x_torch = torch.randn(input_shape, dtype=torch.bfloat16, requires_grad=True)
    b_torch = torch.randn(bias_shape, dtype=torch.bfloat16, requires_grad=True)
    grad_torch = torch.randn(input_shape, dtype=torch.bfloat16)

    out_torch = torch.nn.functional.gelu(x_torch + b_torch, approximate=approximate)
    out_torch.backward(grad_torch)

    x_tt = ttnn.from_torch(x_torch.detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b_torch.detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    grad_a_tt, grad_b_tt = ttnn.bias_gelu_bw(grad_tt, x_tt, b_tt, approximate=approximate)

    assert tuple(grad_a_tt.shape) == input_shape
    assert tuple(grad_b_tt.shape) == bias_shape

    # Deallocating grad_a must not invalidate grad_b when shapes match.
    grad_a_host = ttnn.to_torch(grad_a_tt)
    ttnn.deallocate(grad_a_tt)
    grad_b_host = ttnn.to_torch(grad_b_tt)

    assert torch.allclose(grad_a_host.float(), x_torch.grad.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(grad_b_host.float(), b_torch.grad.float(), rtol=3e-2, atol=3e-2)
