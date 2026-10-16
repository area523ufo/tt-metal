# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("eps", [-1.0, -0.5, 0.0, 0.2, 0.7])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_logiteps_bw_domain_and_negative_eps(device, eps, dtype):
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    vals = torch.tensor(
        [[-0.75, -0.25, 0.0, 0.1, 0.5, 0.9, 1.0, 1.25]] * 4,
        dtype=torch_dtype,
    ).reshape(1, 1, 32, 1).expand(1, 1, 32, 32).contiguous()
    grad = torch.ones_like(vals)

    expected = torch.ops.aten.logit_backward(grad, vals, None if eps < 0.0 else eps)

    x_tt = ttnn.from_torch(vals, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    g_tt = ttnn.from_torch(grad, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    (got_tt,) = ttnn.logiteps_bw(g_tt, x_tt, eps=eps)
    got = ttnn.to_torch(got_tt)

    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    assert torch.equal(torch.isinf(got), torch.isinf(expected))
    finite_mask = torch.isfinite(expected)
    if finite_mask.any():
        assert torch.allclose(got[finite_mask].float(), expected[finite_mask].float(), rtol=1e-2, atol=1e-2)
