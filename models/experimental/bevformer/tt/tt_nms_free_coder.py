# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's NMS-free box coder, from UniAD's ``TtNMSFreeCoder``
(``models/experimental/uniad/tt/ttnn_nms_free_coder.py``).

The top-k selection, the box decoding, the ``post_center_range`` test and upstream
``BEVFormerHead.get_bboxes``' move of z to the box's bottom face run on device, on fixed
``(bs, max_num)`` tensors, with the range test as a validity mask. Only dropping the invalid boxes
runs on host (:meth:`TtNMSFreeCoder.compact`): it leaves a data-dependent number of boxes, which no
device tensor can hold, as ttnn shapes, and a trace's, are fixed. For BEVFormer-base it drops none:
the head scales the centers into ``pc_range``, inside ``post_center_range``.

Upstream's ``score_threshold`` is not ported: no BEVFormer config sets it, and its fallback, which
lowers the threshold by 10% at a time until some score passes, depends on the scores.
"""

import ttnn
from models.experimental.bevformer.model_config import (
    BOX_CENTER,
    BOX_SIZE,
    CODE_COS,
    CODE_H,
    CODE_SIN,
    CODE_VELOCITY,
    CODE_WL,
    CODE_XY,
    CODE_Z,
    MAX_NUM,
    NUM_CLASSES,
    PC_RANGE,
    POST_CENTER_RANGE,
    SCORE_DTYPE,
)

# Candidates the exact float32 top-k ranks; see TtNMSFreeCoder.topk.
NUM_CANDIDATES = 512


def denormalize_bbox(normalized_bboxes):
    """Box predictions (``model_config.py``'s box code, centers in metres) to
    ``(cx, cy, cz, w, l, h, yaw, vx, vy)`` boxes."""
    return ttnn.concat(
        [
            normalized_bboxes[..., CODE_XY],
            normalized_bboxes[..., CODE_Z],
            ttnn.exp(normalized_bboxes[..., CODE_WL]),
            ttnn.exp(normalized_bboxes[..., CODE_H]),
            ttnn.atan2(normalized_bboxes[..., CODE_SIN], normalized_bboxes[..., CODE_COS]),
            normalized_bboxes[..., CODE_VELOCITY],
        ],
        dim=-1,
    )


class TtNMSFreeCoder:
    """BEVFormer's NMS-free box coder with top-k and decoding on device; the arguments are
    ``reference.nms_free_coder.NMSFreeCoder``'s."""

    def __init__(
        self,
        pc_range=PC_RANGE,
        voxel_size=None,
        post_center_range=POST_CENTER_RANGE,
        max_num=MAX_NUM,
        score_threshold=None,
        num_classes=NUM_CLASSES,
    ):
        if score_threshold is not None:
            raise NotImplementedError("TtNMSFreeCoder has no score_threshold; see the module docstring")
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def topk(self, cls_scores, bbox_preds):
        """``(bs, num_query, num_classes)`` ``SCORE_DTYPE`` logits and ``(bs, num_query, code_size)``
        box predictions to each sample's top ``max_num`` scores, sorted, as device tensors: scores
        ``(bs, max_num)`` float32, labels ``(bs, max_num)`` uint32, query indexes ``(bs, max_num, 1)``
        uint32 and boxes ``(bs, max_num, 9)`` in the box predictions' dtype.

        ttnn.topk ranks float32 on a single core, far slower than bfloat16, whose large k takes
        a multi-core route on Blackhole; bfloat16 alone ties many scores. So the logits, which
        rank as their sigmoid scores do, are ranked in three steps:

        1. A bfloat16 top-k: as rounding is monotonic, its k-th value is the true k-th logit
           rounded, the pivot.
        2. A bfloat16 top-``NUM_CANDIDATES`` of the logits minus the pivot: near the k-th logit
           the differences are small, where bfloat16 resolves them finely.
        3. A float32 top-k of the candidates' own logits, returning their original indexes.

        The result is exact unless more than ``NUM_CANDIDATES - max_num`` logits tie with the
        k-th one after the pivot shift in bfloat16.
        """
        if cls_scores.dtype != SCORE_DTYPE:
            raise ValueError(f"cls_scores must be {SCORE_DTYPE}, got {cls_scores.dtype}")
        bs, num_query, num_classes = cls_scores.shape
        assert num_classes == self.num_classes, f"{num_classes} class logits, coder has {self.num_classes}"
        # floor_div below emits float32, exact for indexes up to 2**24.
        assert (
            num_query * num_classes <= 2**24
        ), f"{num_query * num_classes} scores, float32 indexes are exact to 2**24"
        num_candidates = min(NUM_CANDIDATES, num_query * num_classes)
        assert num_candidates >= self.max_num, f"{num_candidates} candidates for a top-{self.max_num}"
        logits = ttnn.reshape(cls_scores, (bs, 1, 1, num_query * num_classes))
        pivot, _ = ttnn.topk(ttnn.typecast(logits, ttnn.bfloat16), k=self.max_num, dim=-1)
        pivot = ttnn.typecast(pivot[..., self.max_num - 1 : self.max_num], SCORE_DTYPE)
        shifted = ttnn.typecast(ttnn.subtract(logits, pivot), ttnn.bfloat16)
        _, candidates = ttnn.topk(shifted, k=num_candidates, dim=-1)
        # A float32 top-k takes uint32 indexes.
        candidates = ttnn.typecast(candidates, ttnn.uint32)
        top_logits, indexes = ttnn.topk(
            ttnn.gather(logits, -1, candidates), k=self.max_num, dim=-1, indices_tensor=candidates
        )
        scores = ttnn.sigmoid(top_logits)
        labels = ttnn.remainder(indexes, num_classes)
        # gather takes uint32 indexes.
        query_index = ttnn.typecast(ttnn.floor_div(indexes, num_classes), ttnn.uint32)
        query_index = ttnn.reshape(query_index, (bs, self.max_num, 1))
        code_index = ttnn.repeat(query_index, (1, 1, bbox_preds.shape[-1]))
        boxes = denormalize_bbox(ttnn.gather(bbox_preds, 1, code_index))
        return (
            ttnn.reshape(scores, (bs, self.max_num)),
            ttnn.reshape(labels, (bs, self.max_num)),
            query_index,
            boxes,
        )

    def valid(self, boxes):
        """``(bs, max_num)`` validity of :meth:`topk`'s boxes, 1.0 or 0.0: the box center inside
        ``post_center_range``. On device."""
        valid = None
        for axis in range(3):
            center = boxes[..., BOX_CENTER.start + axis : BOX_CENTER.start + axis + 1]
            in_range = ttnn.multiply(
                ttnn.ge(center, self.post_center_range[axis]), ttnn.le(center, self.post_center_range[3 + axis])
            )
            valid = in_range if valid is None else ttnn.multiply(valid, in_range)
        return ttnn.reshape(valid, tuple(valid.shape)[:-1])

    @staticmethod
    def bottom_center(boxes):
        """Boxes with z moved from the box's gravity center to its bottom face, as upstream
        ``BEVFormerHead.get_bboxes`` sets it. On device."""
        z = BOX_CENTER.start + 2
        height = BOX_SIZE.start + 2
        bottom = ttnn.subtract(boxes[..., z : z + 1], ttnn.multiply(boxes[..., height : height + 1], 0.5))
        return ttnn.concat([boxes[..., :z], bottom, boxes[..., z + 1 :]], dim=-1)

    def finalize(self, boxes):
        """:meth:`topk`'s boxes to upstream ``get_bboxes``' boxes and their validity, tested before
        the move, as upstream's coder does."""
        return self.bottom_center(boxes), self.valid(boxes)

    @staticmethod
    def compact(scores, labels, boxes, valid):
        """:meth:`finalize`'s device tensors to one ``(boxes, scores, labels)`` per sample on the
        host, the valid ones: the one step whose output size depends on the data."""
        scores, labels, boxes, valid = (ttnn.to_torch(t) for t in (scores, labels, boxes, valid))
        return [
            (b.float()[v.bool()], s.float()[v.bool()], lab.long()[v.bool()])
            for s, lab, b, v in zip(scores, labels, boxes, valid)
        ]

    def decode(self, all_cls_scores, all_bbox_preds):
        """The head's ``(L, bs, num_query, *)`` device outputs to one dict of host boxes, scores and
        labels per sample, from the last decoder layer, as ``reference.nms_free_coder.NMSFreeCoder.decode``
        returns them; all but dropping the invalid boxes on device."""
        scores, labels, _, boxes = self.topk(all_cls_scores[-1], all_bbox_preds[-1])
        return [
            {"bboxes": b, "scores": s, "labels": lab}
            for b, s, lab in self.compact(scores, labels, boxes, self.valid(boxes))
        ]
