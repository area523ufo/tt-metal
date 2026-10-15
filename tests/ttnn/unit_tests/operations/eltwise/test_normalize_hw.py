# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_normalize_hw(x):
    xd = x.double()
    mean = xd.mean(dim=(2, 3), keepdim=True)
    std = xd.std(dim=(2, 3), keepdim=True, unbiased=False)
    return ((xd - mean) / std).float()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scale", [1.0, 100.0, 1e-3])
@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 2, 64, 64]])
def test_normalize_hw(device, dtype, scale, shape):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    x = (torch.randn(shape) * scale).to(torch_dtype)

    got = ttnn.to_torch(ttnn.normalize_hw(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)))

    assert_with_pcc(torch_normalize_hw(x.float()), got.float(), 0.999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_normalize_hw_zero_variance(device, dtype):
    # Every element equal, so the divisor is zero. rsqrt(0) is inf where
    # reciprocal(sqrt(0)) was, and 0 * inf resolves the same way.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = torch.full([1, 1, 32, 32], 3.0, dtype=torch_dtype)

    got = ttnn.to_torch(ttnn.normalize_hw(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)))

    assert torch.isfinite(got.float()).all() or torch.isnan(got.float()).all()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [[1, 1, 4, 4], [1, 1, 17, 50], [2, 3, 7, 19]])
def test_var_std_normalize_hw_non_tile_aligned(device, dtype, shape):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(42)
    x = (torch.randn(shape) * 2.0 + 1.0).to(torch_dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    ref_var = x.float().var(dim=(2, 3), keepdim=True, unbiased=False)
    ref_std = x.float().std(dim=(2, 3), keepdim=True, unbiased=False)
    ref_norm = torch_normalize_hw(x.float())

    got_var = ttnn.to_torch(ttnn.var_hw(xt)).float()
    got_std = ttnn.to_torch(ttnn.std_hw(xt)).float()
    got_norm = ttnn.to_torch(ttnn.normalize_hw(xt)).float()

    atol = 5e-2 if dtype == ttnn.bfloat16 else 1e-4
    rtol = 5e-2 if dtype == ttnn.bfloat16 else 1e-4
    assert torch.allclose(got_var, ref_var, rtol=rtol, atol=atol)
    assert torch.allclose(got_std, ref_std, rtol=rtol, atol=atol)
    assert torch.allclose(got_norm, ref_norm, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [[1, 1, 4, 4], [1, 2, 7, 19]])
def test_normalize_global_non_tile_aligned(device, dtype, shape):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(7)
    x = (torch.randn(shape) * 3.0 - 0.5).to(torch_dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    xd = x.double()
    ref_global = ((xd - xd.mean()) / xd.std(unbiased=False)).float()
    got_t = ttnn.normalize_global(xt)
    assert list(got_t.logical_shape) == shape
    got_global = ttnn.to_torch(got_t).float()

    atol = 5e-2 if dtype == ttnn.bfloat16 else 1e-4
    rtol = 5e-2 if dtype == ttnn.bfloat16 else 1e-4
    assert torch.allclose(got_global, ref_global, rtol=rtol, atol=atol)

