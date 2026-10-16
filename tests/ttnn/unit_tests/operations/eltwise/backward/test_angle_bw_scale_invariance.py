# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)


def _tensor(val, dtype, torch_dtype, device):
    return ttnn.from_torch(
        torch.full(SHAPE, val, dtype=torch_dtype),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


@pytest.mark.parametrize(
    "re_val, im_val, grad_val, expected_re, expected_im",
    [
        # Large inputs (|z| > 2^64): x^2 + y^2 overflows FP32/BF16, but true gradient is finite
        (1e20, 1e20, 1e20, -0.5, 0.5),
        (0.0, 2e20, 2e20, -1.0, 0.0),
        # Small inputs (0 < |z| < 2^-63): x^2 + y^2 underflows FP32/BF16 to 0, but true gradient is finite
        (0.0, 1e-20, 1.0, -1e20, 0.0),
        (1e-20, 0.0, 1.0, 0.0, 1e20),
        # Zero upstream gradient with small inputs must not form 0 * inf = NaN
        (1e-20, 1e-20, 0.0, 0.0, 0.0),
        # Origin (0, 0) returns (0, 0)
        (0.0, 0.0, 1.0, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
def test_angle_bw_extreme_scales(device, re_val, im_val, grad_val, expected_re, expected_im, dtype, torch_dtype):
    inp = ttnn.complex_tensor(
        _tensor(re_val, dtype, torch_dtype, device),
        _tensor(im_val, dtype, torch_dtype, device),
    )
    grad = _tensor(grad_val, dtype, torch_dtype, device)
    out = ttnn.angle_bw(grad, inp, memory_config=ttnn.DRAM_MEMORY_CONFIG)[0]

    got_re = ttnn.to_torch(out.real).float().flatten()[0].item()
    got_im = ttnn.to_torch(out.imag).float().flatten()[0].item()

    assert not math.isnan(got_re) and not math.isnan(got_im), f"Unexpected NaN: ({got_re}, {got_im})"
    tol = 0.05 if dtype == ttnn.bfloat16 else 1e-4
    assert abs(got_re - expected_re) <= tol * max(1.0, abs(expected_re)), f"real {got_re} vs {expected_re}"
    assert abs(got_im - expected_im) <= tol * max(1.0, abs(expected_im)), f"imag {got_im} vs {expected_im}"
