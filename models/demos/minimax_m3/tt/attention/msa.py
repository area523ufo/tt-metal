# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 MSA (sparse) attention — the real model forward for the sparse layers (3-59).

Unlike the dense path (ring_joint, which reads the KV cache + gathers across SP *internally*),
``sparse_sdpa_msa`` is a pure dense-context kernel: it takes full-length K/V tensors and has no
cache-read. So the cache + cross-device gather live in THIS wrapper:

  each SP device holds a sequence shard of K / V / index_k (from the chunked-KV cache read).
  We AllGather those across the SP axis so every device materialises the full context, then:

    indexer_score_msa(index_q, index_k_full, chunk_start_idx=cached_len)  -> block scores
    topk_large_indices(topk_blocks)                                      -> block-ids   (the op
                                              already force-locals the current block, +inf)
    sparse_sdpa_msa(q, k_full, v_full, block-ids)                        -> attention out

SP sharding is CONTIGUOUS, no zigzag/balancing (per the op authors: chunked prefill needs no causal
load-balancing — MSA work per query is a fixed top-k, not the dense causal triangle).

``chunk_start_idx`` is the global position of query row 0. Because the indexer scores over the
gathered full context and the current chunk's queries all begin at ``cached_len``, this scalar is
uniform across SP devices (no per-device offset needed). Causality is encoded entirely by the block
selection; sparse_sdpa_msa applies no token mask.
"""

from loguru import logger

import ttnn
from models.demos.minimax_m3.utils.profiler_utils import FINE, zone

from .operations import apply_qk_norm_per_head, apply_rope

# sparse_sdpa_msa's per-core L1 block cache for the gathered K/V (as many slots as fit in free L1). On: at small SP
# consecutive queries on a core re-select mostly the same blocks, so a resident block saves a DRAM re-read. Output
# is identical either way. Not the model's DRAM KV cache.
MSA_KV_BLOCK_CACHE = True

# Packed forwards (prefill_segments): gather the next warm segment's K/V/index_k prefix on a strip of cores while
# this segment's block selection and sparse_sdpa_msa run on the rest (CCLManager.get_msa_overlap). Output is
# identical to the serial chain. Needs at least two warm segments in the forward.
MSA_SEGMENT_OVERLAP = True

# indexer_score_msa's program config and the banded schedule it implies (indexer_score_work_split.hpp): the op
# takes no core grid, so the overlap window can only hold it if its rows fit the compute sub-device.
INDEXER_Q_CHUNK, INDEXER_K_CHUNK = 64, 1024


def _ensure_dram(t):
    """high_bw_all_gather streams its source from DRAM; move an L1-resident activation there first."""
    if t.memory_config().buffer_type != ttnn.BufferType.DRAM:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def high_bw_sp_gather(
    t, mesh_config, ccl_manager, out_buf, *, input_batch_index=None, gathered_dim_size=None, overlap=None
):
    """SP all-gather of ``t`` on dim 2 with ``ttnn.experimental.high_bw_all_gather`` into the persistent
    ``out_buf`` (CCLManager.get_high_bw_gather_buffer); the returned tensor aliases it, do not deallocate.
    ``input_batch_index`` selects one slot of a multi-slot [B, 1, rows, D] cache; ``gathered_dim_size``
    bounds the rows moved per rank (rank r still lands at its fixed slot r*rows). ``overlap`` (an
    MsaOverlap, see tt/ccl.py) confines the workers to its gather sub-device with its persistent semaphores,
    so the gather can run beside compute on the other sub-device."""
    kwargs = {}
    if overlap is not None:
        kwargs = dict(
            subdevice_id=overlap.gather_sd_id,
            sub_core_grids=overlap.gather_cores,
            ready_semaphore=overlap.ready_semaphore,
            data_valid_semaphore=overlap.data_valid_semaphore,
        )
    return ttnn.experimental.high_bw_all_gather(
        t,
        dim=2,
        output_tensor=out_buf,
        cluster_axis=mesh_config.sp_axis,
        num_links=ccl_manager.num_links,
        input_batch_index=input_batch_index,
        gathered_dim_size=gathered_dim_size,
        **kwargs,
    )


def _split_index_heads(t, index_dim):
    """[1, 1, S, n*index_dim] -> [1, n, S, index_dim] (head-major split, like the main QKV split)."""
    s = t.shape[2]
    n = t.shape[-1] // index_dim
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [1, s, n, index_dim])
    t = ttnn.permute(t, (0, 2, 1, 3))  # [1, n, S, index_dim]
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def index_branch_forward(
    hidden_states,
    weights,
    rope_mats,
    transformation_mat,
    *,
    index_dim,
    rms_norm_eps,
    kv_actual_global=None,
    cluster_axis=None,
):
    """The MSA index branch: pre-roped index_q (n index heads, 1/TP col) + index_k (single shared head).

    proj -> split heads -> per-head RMSNorm -> RoPE. index_q_proj is column-parallel (4 index heads ->
    1/TP col); index_k_proj is replicated (shared head). Per device this yields index_q [1, n_idx_local,
    S, index_dim] (n_idx_local=1 at TP=4) and index_k [1, 1, S, index_dim] -> the MSA indexer.

    VERIFIED 2026-06-26 against transformers-main MiniMaxM3VLIndexer source (not just a summary):
      * order proj -> q_norm/k_norm -> apply_rotary_pos_emb — matches.
      * rope on BOTH index_q AND index_k — matches.
      * rotary width: reference does apply_rotary_pos_emb(idx_q, idx_k, cos[..,:index_head_dim], ...);
        with head_dim=128, partial_rotary_factor=0.5 the model cos/sin are 64-wide, and index_head_dim=
        sparse_index_dim=128, so the slice yields the full 64 -> PARTIAL-64 (rotate first 64 of the
        128-wide index head), same as main attention. Our apply_rope(rope_mats=main 64-wide) matches.
      * norm: index_q_norm/index_k_norm gains ship in the checkpoint, applied per-head.
    """
    iq = _split_index_heads(
        ttnn.linear(hidden_states, weights.index_q_proj), index_dim
    )  # [1, n_idx_local, S, index_dim]
    iq = apply_qk_norm_per_head(iq, weights.index_q_norm, rms_norm_eps)
    iq = apply_rope(
        iq,
        rope_mats,
        transformation_mat,
        is_decode_mode=False,
        kv_actual_global=kv_actual_global,
        cluster_axis=cluster_axis,
    )

    ik = _split_index_heads(
        ttnn.linear(hidden_states, weights.index_k_proj), index_dim
    )  # [1, 1, S, index_dim] (shared)
    ik = apply_qk_norm_per_head(ik, weights.index_k_norm, rms_norm_eps)
    ik = apply_rope(
        ik,
        rope_mats,
        transformation_mat,
        is_decode_mode=False,
        kv_actual_global=kv_actual_global,
        cluster_axis=cluster_axis,
    )
    return iq, ik


def msa_select_blocks(
    index_q,
    index_k,
    *,
    chunk_start_idx,
    scale,
    num_groups,
    block_size,
    topk_blocks,
    cluster_axis=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    kv_len=None,
    overlap=None,
):
    """Indexer scores + top-k -> block ids [1, num_groups, Sq, topk] (uint32, row-major). Arguments as for
    msa_indexer_sparse. ``overlap`` confines the top-k to the compute sub-device (the indexer needs no
    confinement: its banded schedule stays inside the compute rows, see indexer_rows_used)."""
    # Block scores: scaled dot, causal -inf for future, group-sum, block-max-pool. bf16 row-major out.
    with zone("indexer"):
        block_scores = ttnn.experimental.indexer_score_msa(
            index_q,
            index_k,
            chunk_start_idx=chunk_start_idx,
            scale=scale,
            num_groups=num_groups,
            block_size=block_size,
            program_config=ttnn.IndexerScoreProgramConfig(
                q_chunk_size=INDEXER_Q_CHUNK, k_chunk_size=INDEXER_K_CHUNK, head_group_size=0
            ),
            seq_shard_axes=[cluster_axis] if cluster_axis is not None else [],
            block_cyclic_sp_axis=block_cyclic_sp_axis,
            block_cyclic_chunk_local=block_cyclic_chunk_local,
            kv_len=kv_len,
        )

        # Top-k block ids (uint32 row-major) — the block selection that encodes causality. The op already
        # force-locals the current (diagonal) block; upstream minimax_m3_vl forces ONLY the local block.
        # valid_length bounds the search to the populated block columns when K sits in a worst-case buffer.
        topk_kwargs = {}
        if kv_len is not None:
            assert kv_len % block_size == 0, f"kv_len ({kv_len}) must be a whole number of {block_size}-token blocks"
            topk_kwargs["valid_length"] = kv_len // block_size
        if overlap is not None:
            topk_kwargs["subdevice_id"] = overlap.compute_sd_id
            topk_kwargs["sub_core_grids"] = overlap.compute_cores
        block_ids = ttnn.experimental.topk_large_indices(block_scores, k=topk_blocks, **topk_kwargs)
        if _block_id_sink is not None:
            _block_id_sink(block_ids)
    return block_ids


def msa_sparse_attend(
    q_rm,
    k,
    v,
    block_ids,
    *,
    chunk_start_idx,
    scale,
    block_size,
    cluster_axis=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    overlap=None,
):
    """sparse_sdpa_msa over the selected blocks: row-major q in, row-major out [1, Hq, Sq, head_dim]. The caller
    owns the layout conversions (the overlap window admits only ops that stay on one sub-device)."""
    # sparse_sdpa_msa (#48700): q + block-ids row-major, K/V tiled; expands blocks->tokens internally.
    # chunk_start_idx + cluster_axis drive the token-level diagonal-block causal mask with the per-device
    # SP start (chunk_start = chunk_start_idx + rank*Sq); q must be bf16 (the op rejects fp8 q under causal).
    return ttnn.transformer.sparse_sdpa_msa(
        q_rm,
        k,
        v,
        block_ids,
        scale=scale,
        block_size=block_size,
        chunk_start_idx=chunk_start_idx,
        cluster_axis=cluster_axis,
        block_cyclic_sp_axis=block_cyclic_sp_axis,
        block_cyclic_chunk_local=block_cyclic_chunk_local,
        enable_kv_block_cache=MSA_KV_BLOCK_CACHE,
        core_grid=overlap.compute_cores if overlap is not None else None,
    )


def msa_indexer_sparse(
    index_q,
    index_k,
    q,
    k,
    v,
    *,
    chunk_start_idx,
    scale,
    num_groups,
    block_size,
    topk_blocks,
    device,
    return_block_ids=False,
    cluster_axis=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    kv_len=None,
):
    """The MSA op chain over a FULL-context (already-gathered) K/V; index_q/q may stay SP-sharded.

    index_q [1, num_groups, Sq, index_dim]   index_k [1, 1, T, index_dim]   (1 shared index-k head)
    q       [1, Hq, Sq, head_dim]            k, v    [1, n_kv, T, head_dim]  (TILE layout)
    cluster_axis: when set, the merged op derives a PER-DEVICE causal chunk_start from the device's
      mesh coordinate along that axis -> chunk_start = chunk_start_idx + rank*Sq (Sq = q's S/sp rows),
      so q/index_q stay SP-sharded. None -> uniform chunk_start_idx (single-device / gathered query).
      (Replaces the old host-built per-device chunk_offset tile; mesh-coord approach, #47939.)
    block_cyclic_sp_axis / block_cyclic_chunk_local (set together): the AllGather'd K/V/index_k are
      still in block-cyclic ("slab") SP order; the indexer (K reader) and sparse_sdpa_msa (K reader /
      V writer) remap each natural block-id to its physical block IN-KERNEL (#49490/#48772), so the
      caller no longer untilize->transpose->tilize them to natural order on the host. None -> the
      tensors are already natural (single-chunk / pre-reordered).
    kv_len: when K/index_k are allocated at a FIXED worst-case T (the persistent high_bw_all_gather
      buffers, T = cache capacity) this is the valid natural-position prefix actually written (a multiple
      of block_size). The indexer scores/writes only columns [0, kv_len) (hash-excluded runtime arg, so a
      growing prefix reuses one program) and top-k ranks only those kv_len/block_size block columns; the
      stale tail past kv_len is never read. None -> T is the valid length (the legacy exact-size gather).
    -> out  [1, Hq, Sq, head_dim]
    """
    geometry = dict(
        chunk_start_idx=chunk_start_idx,
        cluster_axis=cluster_axis,
        block_cyclic_sp_axis=block_cyclic_sp_axis,
        block_cyclic_chunk_local=block_cyclic_chunk_local,
    )
    block_ids = msa_select_blocks(
        index_q,
        index_k,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
        topk_blocks=topk_blocks,
        kv_len=kv_len,
        **geometry,
    )
    with zone("sparse_sdpa"):
        out = msa_sparse_attend(
            ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT), k, v, block_ids, scale=scale, block_size=block_size, **geometry
        )
        # sparse_sdpa_msa returns ROW_MAJOR; the model's concat_heads (prefill.py) needs TILE — match the
        # dense (ring_joint) output so the shared post-attention path works for MSA layers too.
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    return (out, block_ids) if return_block_ids else out


# Debug hook (default off): called with every MSA call's top-k block ids, e.g. to compare block selection
# between two runs. Set with set_block_id_sink(fn) / set_block_id_sink(None).
_block_id_sink = None


def set_block_id_sink(fn):
    global _block_id_sink
    _block_id_sink = fn


def msa_sp_attention_nocache(
    q,
    k,
    v,
    index_q,
    index_k,
    *,
    mesh_config,
    ccl_manager,
    cached_len,
    s_local,
    scale,
    block_size,
    topk_blocks,
    num_groups=1,
    return_block_ids=False,
):
    """Sharded-query MSA under SP: AllGather only the KEYS; q/index_q stay sharded (S/sp rows/device).

    Each device scores ONLY its own S/sp query rows against the gathered full context, with per-device
    causality from the op's native mesh-coord chunk_start (cluster_axis=sp_axis -> rank*s_local on top of
    chunk_start_idx=cached_len). Output stays SP-sharded [1, Hq, s_local, head_dim] — no replication, no
    reshard — which is what the SP residual stream needs. This is the deployed path (vs the gather-everything
    golden, which gathers the query too). index_q is the device's group's index head; q is its TP head-slice.
    """
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device

    sp = device.shape[sp_axis]
    if sp == 1:
        # Nothing to gather: one SP row already holds the full context (the collective rejects a one-device axis).
        def gather(key, t):
            return t

    else:
        # Exact-size activations: the worst-case buffer IS the gathered shape, so no kv_len bound is needed.
        def gather(key, t):
            src = _ensure_dram(t)
            buf = ccl_manager.get_high_bw_gather_buffer(
                key, (src.shape[0], src.shape[1], src.shape[2] * sp, src.shape[3]), src.dtype, src.layout
            )
            full = high_bw_sp_gather(src, mesh_config, ccl_manager, buf)
            if src is not t:
                src.deallocate(True)  # the DRAM staging copy; the caller still owns `t`
            return full

    # ag_kv feeds sparse_sdpa_msa; ag_index_k feeds the indexer — split so the two AG costs are
    # attributable to their consumer (see tests/perf/profile_prefill.py).
    with zone("ag_kv"):
        k_full = gather("msa_nocache_k", k)
        v_full = gather("msa_nocache_v", v)
    with zone("ag_index_k"):
        index_k_full = gather("msa_nocache_index_k", index_k)
    # Per-device causality via the merged op's native mesh-coord chunk_start (#47939): device r derives
    # chunk_start = cached_len + r*Sq (Sq = q's s_local rows) from its coordinate along cluster_axis=sp_axis.
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
        topk_blocks=topk_blocks,
        device=device,
        cluster_axis=sp_axis,
    )


def msa_cache_read_extent(cached_len, chunk_local, sp, block_size):
    """(kv_len, n_rows) for the cross-chunk MSA read after the chunk at ``cached_len`` has been written.

    The KV writer places global position g on SP rank (g // chunk_local) % sp, so a chunk that starts
    mid-slab (``cached_len`` not a whole number of ``chunk_local * sp`` chunks, e.g. a multi-turn resume at a
    32-token boundary) leaves the ranks unevenly filled. ``n_rows`` is the FULLEST rank's local row count
    covering the written prefix [0, cached_len + chunk_global) -- rank 0, which owns the first block of every
    slab -- since ``high_bw_all_gather`` gathers the same local prefix from every rank. ``kv_len`` is that prefix
    rounded up to whole ``block_size`` blocks (the indexer pools and top-k selects in blocks). Positions past a
    rank's written prefix -- including [end, kv_len) -- hold zeros or stale finite KV in the persistent gather
    buffer (zeroed at allocation, see CCLManager.get_high_bw_gather_buffer), all future to every query of the
    chunk, so the indexer's causal mask and sparse_sdpa_msa's diagonal-block mask never attend them.
    """
    assert (
        cached_len % ttnn.TILE_SIZE == 0
    ), f"cached_len={cached_len} must be a multiple of {ttnn.TILE_SIZE} (the KV writer's tile grid)"
    assert chunk_local % block_size == 0, f"chunk_local={chunk_local} must be a whole number of {block_size} blocks"
    chunk_global = chunk_local * sp
    end = cached_len + chunk_global  # the chunk (incl. its pad tail) is written up to here
    kv_len = (end + block_size - 1) // block_size * block_size
    full_slabs, rem = divmod(end, chunk_global)
    n_rows = full_slabs * chunk_local + min(rem, chunk_local)
    return kv_len, n_rows


def msa_cache_gather(
    kv_cache, *, slot, mesh_config, ccl_manager, cached_len, chunk_local, block_size, set_idx=0, overlap=None
):
    """Gather (user, layer) ``slot`` of the packed cache across SP into the persistent worst-case buffers:
    ``(k_full, v_full, index_k_full, kv_len)``. ``set_idx`` selects one of two buffer sets so a gather can
    land while the other set is still being read (the packed-forward overlap); ``overlap`` runs it on the
    gather sub-device. The gathered tensors alias the persistent buffers: never deallocate."""
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device
    sp = device.shape[sp_axis]
    assert sp > 1, "msa_cache_gather needs sp > 1 (high_bw_all_gather rejects a one-device axis)"
    # The replicated cache tensors carry a 1-D tensor topology, and high_bw_all_gather requires
    # cluster_axis < that rank, so the SP gather only works with SP on mesh axis 0.
    assert sp_axis == 0, f"msa_cache_gather needs sp_axis == 0 (got {sp_axis})"
    seq_local = kv_cache.k.shape[2]  # per-device cache capacity (rows)
    kv_len, n_rows = msa_cache_read_extent(cached_len, chunk_local, sp, block_size)
    assert (
        n_rows <= seq_local and kv_len <= seq_local * sp
    ), f"cache read past capacity: {n_rows} rows / kv_len {kv_len} > {seq_local} rows x {sp}"

    def gather(key, cache_t):
        buf = ccl_manager.get_high_bw_gather_buffer(
            msa_gather_buffer_key(key, set_idx), (1, 1, seq_local * sp, cache_t.shape[3]), cache_t.dtype
        )
        return high_bw_sp_gather(
            cache_t,
            mesh_config,
            ccl_manager,
            buf,
            input_batch_index=slot,
            gathered_dim_size=n_rows * sp,
            overlap=overlap,
        )

    # ag_kv feeds sparse_sdpa_msa; ag_index_k feeds the indexer — split so the two AG costs are
    # attributable to their consumer (see tests/perf/profile_prefill.py).
    with zone("ag_kv"):
        k_full = gather("msa_cache_k", kv_cache.k)
        v_full = gather("msa_cache_v", kv_cache.v)
    with zone("ag_index_k"):
        index_k_full = gather("msa_cache_index_k", kv_cache.index_k)
    return k_full, v_full, index_k_full, kv_len


def msa_gather_buffer_key(key, set_idx):
    """Set 0 keeps the serial path's key, so both paths share those buffers."""
    return key if set_idx == 0 else f"{key}/{set_idx}"


def msa_sp_attention_cache_read(
    q,
    index_q,
    kv_cache,
    *,
    slot,
    mesh_config,
    ccl_manager,
    cached_len,
    chunk_local,
    scale,
    block_size,
    topk_blocks,
    num_groups=1,
):
    """Cross-chunk MSA: the current chunk's queries attend the accumulated context, gathered across SP
    straight out of (user, layer) ``slot`` of the packed ``[num_users*num_layers, 1, seq_local, hd]``
    ND-sharded cache by ``high_bw_all_gather`` (``input_batch_index=slot``, ``gathered_dim_size`` = the
    written prefix, incl. the current chunk that prefill.py wrote before calling us). The output is the
    persistent worst-case buffer ``[1, 1, seq_local*sp, hd]`` with rank r at the fixed slot r*seq_local,
    which is the block-cyclic layout the indexer / sparse_sdpa_msa decode in-kernel (stride T/sp ==
    seq_local); ``kv_len`` bounds them to the written prefix. Returns the chunk's SP-sharded attention out
    ``[1, Hq_local, s_local, hd]``. The gathered tensors alias the persistent buffers: never deallocate.
    """
    sp_axis = mesh_config.sp_axis
    k_full, v_full, index_k_full, kv_len = msa_cache_gather(
        kv_cache,
        slot=slot,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        cached_len=cached_len,
        chunk_local=chunk_local,
        block_size=block_size,
    )
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
        topk_blocks=topk_blocks,
        device=ccl_manager.mesh_device,
        cluster_axis=sp_axis,
        block_cyclic_sp_axis=sp_axis,
        block_cyclic_chunk_local=chunk_local,
        kv_len=kv_len,
    )


def indexer_rows_used(sq, t, grid_x, grid_y):
    """Grid rows indexer_score_msa lights up for Sq query rows against T keys (its banded schedule,
    indexer_score_work_split.hpp, with INDEXER_Q_CHUNK / INDEXER_K_CHUNK): groups of q-chunks go to rows, bands
    of k-chunks to columns, and idle rows are filled by replicating groups. Kept in step with the op."""
    groups = max(1, sq // INDEXER_Q_CHUNK)
    bands = -(-t // INDEXER_K_CHUNK)
    group_rows = max(d for d in range(1, min(groups, grid_y) + 1) if groups % d == 0)
    cols = min(bands, grid_x)
    blocks = max(1, min(grid_y // group_rows, bands // cols))
    return group_rows * blocks


_overlap_refusal_logged = set()


def msa_segments_overlap_possible(segments, kv_cache, ccl_manager):
    """Whether the packed forward's warm segments can run the gather/compute overlap: at least two of them, an
    SP axis to gather over, and an indexer schedule that stays inside the compute sub-device."""
    warm = [s for s in segments if s.cached_len > 0]
    if len(warm) < 2:
        return False
    device = ccl_manager.mesh_device
    if device.shape[0] < 2:
        return False
    overlap = ccl_manager.get_msa_overlap()
    sq = warm[0].q.shape[2]
    rows = indexer_rows_used(sq, kv_cache.k.shape[2] * device.shape[0], *overlap.grid)
    if rows > overlap.compute_rows:
        key = (sq, rows)
        if key not in _overlap_refusal_logged:
            _overlap_refusal_logged.add(key)
            logger.warning(
                f"MSA segment overlap off: indexer_score_msa needs {rows} grid rows at Sq={sq}, the compute "
                f"sub-device has {overlap.compute_rows}; the segments run serially"
            )
        return False
    return True


def msa_segments_pipelined(segments, *, kv_cache, mesh_config, ccl_manager, scale, block_size, topk_blocks, num_groups):
    """The warm segments' MSA chains with the next segment's cache gather overlapped under this segment's
    block selection and sparse attention. ``segments``: objects with ``q`` (tiled), ``index_q``, ``slot``,
    ``cached_len``, ``s_local``, in forward order. Returns the row-major attention outputs in that order.

    Schedule, with G(i) = segment i's K/V/index_k gather (gather sub-device, buffer set i % 2) and C(i) = its
    indexer -> top-k -> sparse_sdpa_msa (compute sub-device):
        G(0); for i: join(G(i), C(i-1)); C(i); G(i+1)
    The join is a device-side event over both sub-devices: C(i) then starts once its gather has landed and
    C(i-1) has released the other buffer set, while G(i+1) only waits for G(i) (a program stalls on its own
    sub-device's predecessors). Everything that is not one of those ops -- layout changes, allocations that
    run as programs, deallocations of tensors in flight -- stays outside the window.
    """
    overlap = ccl_manager.get_msa_overlap()
    mesh = ccl_manager.mesh_device
    sp_axis = mesh_config.sp_axis
    # Both buffer sets exist before the window: get_high_bw_gather_buffer allocates with a full-grid op.
    seq_local = kv_cache.k.shape[2]
    for set_idx in (0, 1):
        for key, cache_t in (
            ("msa_cache_k", kv_cache.k),
            ("msa_cache_v", kv_cache.v),
            ("msa_cache_index_k", kv_cache.index_k),
        ):
            ccl_manager.get_high_bw_gather_buffer(
                msa_gather_buffer_key(key, set_idx),
                (1, 1, seq_local * mesh.shape[sp_axis], cache_t.shape[3]),
                cache_t.dtype,
            )
    q_rm = [ttnn.to_layout(s.q, ttnn.ROW_MAJOR_LAYOUT) for s in segments]

    def gather(i):
        s = segments[i]
        with zone(f"gather{i}", FINE):
            return msa_cache_gather(
                kv_cache,
                slot=s.slot,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                cached_len=s.cached_len,
                chunk_local=s.s_local,
                block_size=block_size,
                set_idx=i % 2,
                overlap=overlap,
            )

    outs = []
    keep_alive = []
    try:
        mesh.load_sub_device_manager(overlap.manager_id)
        overlap.loaded = True
        pending = gather(0)
        for i, s in enumerate(segments):
            ttnn.record_event(mesh)
            k_full, v_full, index_k_full, kv_len = pending
            geometry = dict(
                chunk_start_idx=s.cached_len,
                cluster_axis=sp_axis,
                block_cyclic_sp_axis=sp_axis,
                block_cyclic_chunk_local=s.s_local,
            )
            with zone(f"seg{i}", FINE):
                block_ids = msa_select_blocks(
                    s.index_q,
                    index_k_full,
                    scale=scale,
                    num_groups=num_groups,
                    block_size=block_size,
                    topk_blocks=topk_blocks,
                    kv_len=kv_len,
                    overlap=overlap,
                    **geometry,
                )
                with zone("sparse_sdpa"):
                    outs.append(
                        msa_sparse_attend(
                            q_rm[i],
                            k_full,
                            v_full,
                            block_ids,
                            scale=scale,
                            block_size=block_size,
                            overlap=overlap,
                            **geometry,
                        )
                    )
            keep_alive.append(block_ids)
            if i + 1 < len(segments):
                pending = gather(i + 1)
    finally:
        # The join point: clearing the manager waits on both sub-devices before resetting.
        if overlap.loaded:
            mesh.clear_loaded_sub_device_manager()
            overlap.loaded = False
        for t in keep_alive + q_rm:
            ttnn.deallocate(t)
    return outs
