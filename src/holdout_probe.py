"""
Recover held-out points' embeddings from a handful of probe distances,
then check whether the recovered embeddings generalize.

split_pairs (src/datasets.py) only ever withholds pairs among points that
are all still trained via their many other pairs - it can't tell whether
the model learned a real geometry or just memorized/interpolated known
pairwise distances. split_points + make_holdout_point_pairs (also in
src/datasets.py) instead withhold whole points from every training pair,
so their embedding-table rows sit untouched at random init through all of
main training. This module is what happens after training: fit each
held-out point's embedding from a few true probe distances to normally-
trained points, then evaluate its predicted distances to everything else.

Recovery mechanism: each held-out point's embedding is a standalone
nn.Parameter - the only thing the probe-phase AdamW optimizer ever sees.
On every step it is spliced into a frozen, detached copy of the trained
embedding table via torch.Tensor.index_copy to build the table the
forward pass actually reads; gradient flows back only into that
parameter, since the base table has no grad history. Non-holdout rows are
therefore never inside the optimizer's parameter list at all - not
masked, not restored, structurally untouchable by AdamW's weight decay or
anything else - so this really is "fine-tune the new point the same way
training fine-tunes everything else," just scoped to one (or a few) rows.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets import (
    make_distance_examples_from_pairs,
    make_same_triangle_examples_from_pairs,
)
from src.eval_utils import (
    evaluate_distance_examples,
    evaluate_same_triangle_examples,
)

# Task name -> (make_examples_from_pairs, evaluate) used for *post-recovery*
# evaluation only (see eval_tasks below). Recovery itself always fits
# holdout_param against distance probes, regardless of eval_tasks - kept
# separate from src/train_manifold.py's own per-task registry (which also
# needs each task's training-example builder, loss, and forward method) to
# avoid a circular import, since train_manifold imports this module.
_EVAL_BUILDERS = {
    "distance": (make_distance_examples_from_pairs, evaluate_distance_examples),
    "same_triangle": (
        make_same_triangle_examples_from_pairs, evaluate_same_triangle_examples,
    ),
}


def recover_and_evaluate_holdout_points(
    model,
    world,
    holdout_point_pairs,
    probe_steps,
    probe_lr,
    probe_weight_decay,
    batch_size,
    device,
    eval_tasks=("distance",),
):
    """
    Recover every holdout point's embedding from its probe pairs, commit
    the recovered rows into model.emb.weight, then evaluate:

        per_point / aggregate:   each holdout point's predictions on
                                  normally-trained points (eval_pairs),
                                  pooled across holdout points too.
        holdout_to_holdout:      predictions BETWEEN pairs of recovered
                                  holdout points - checks whether
                                  independently-recovered embeddings are
                                  mutually consistent, not just
                                  individually consistent with the
                                  trained points used as probes. None
                                  when fewer than 2 holdout points are
                                  given.

    Recovery (fitting holdout_param) always uses distance probe pairs,
    regardless of eval_tasks - "tuned only on distance." eval_tasks (an
    iterable of keys into _EVAL_BUILDERS, default ("distance",)) instead
    controls which task(s) the *recovered* embeddings are then evaluated
    on: e.g. eval_tasks=("distance", "same_triangle") checks whether an
    embedding recovered purely from distance probes also generalizes to
    a same_triangle head it was never directly probed against. Each of
    per_point/aggregate/holdout_to_holdout becomes a dict keyed by task
    name.

    Model mutation contract: this permanently overwrites the embedding
    rows named in holdout_point_pairs with their recovered values, and
    touches nothing else - no other parameter, no requires_grad flag
    anywhere (this function never calls requires_grad_ on anything; the
    base model's parameters are never added to the optimizer). Safe to
    call as the last step before torch.save(model.state_dict()).
    """

    holdout_indices = sorted(holdout_point_pairs.keys())
    holdout_idx_tensor = torch.tensor(
        holdout_indices, dtype=torch.long, device=device,
    )

    # Never touched again after this point - no grad history, so
    # gradients computed against `table` below flow only into
    # holdout_param, never back into this frozen base.
    base_frozen = model.emb.weight.detach().clone()

    # The only trainable tensor in this whole routine.
    holdout_param = nn.Parameter(base_frozen[holdout_idx_tensor].clone())

    # One fixed batch: every holdout point's probe pairs, pooled.
    all_probe_pairs = np.concatenate(
        [pairs["probe_pairs"] for pairs in holdout_point_pairs.values()],
        axis=0,
    )
    probe_examples = make_distance_examples_from_pairs(
        world, all_probe_pairs,
    )
    probe_i = torch.tensor(
        [example["indices"][0] for example in probe_examples],
        dtype=torch.long, device=device,
    )
    probe_j = torch.tensor(
        [example["indices"][1] for example in probe_examples],
        dtype=torch.long, device=device,
    )
    probe_y = torch.tensor(
        [example["answer"] for example in probe_examples],
        dtype=torch.float32, device=device,
    )

    # Real AdamW, scoped to a parameter that structurally contains only
    # holdout rows - there is nothing else here for its weight decay to
    # shrink.
    optimizer = torch.optim.AdamW(
        [holdout_param], lr=probe_lr, weight_decay=probe_weight_decay,
    )

    model.eval()

    for _ in range(probe_steps):
        # Splice the current holdout values into the frozen base table.
        # Differentiable w.r.t. holdout_param only.
        table = base_frozen.index_copy(0, holdout_idx_tensor, holdout_param)

        prediction = model.forward_distance_from_embeddings(
            F.embedding(probe_i, table),
            F.embedding(probe_j, table),
        )

        loss = F.smooth_l1_loss(prediction, probe_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Commit the recovered rows into the real model. Nothing else was
    # ever touched, so this is the only write to model.emb.weight this
    # function performs.
    with torch.no_grad():
        model.emb.weight.data[holdout_idx_tensor] = holdout_param.detach()

    # Each holdout point's predictions on normally-trained points, now
    # readable via the model's ordinary forward_* methods since the
    # recovered rows are already committed above - once per requested
    # eval task.
    all_eval_pairs = np.concatenate(
        [pairs["eval_pairs"] for pairs in holdout_point_pairs.values()],
        axis=0,
    )

    holdout_pairs = None
    if len(holdout_indices) >= 2:
        holdout_pairs = np.array(
            [
                (h1, h2)
                for idx1, h1 in enumerate(holdout_indices)
                for h2 in holdout_indices[idx1 + 1:]
            ],
            dtype=np.int64,
        )

    per_point = {}
    aggregate = {}
    holdout_to_holdout = {}

    for task in eval_tasks:
        make_examples_from_pairs, evaluate = _EVAL_BUILDERS[task]

        per_point[task] = {
            h: evaluate(
                model,
                make_examples_from_pairs(world, pairs["eval_pairs"]),
                device, batch_size,
            )
            for h, pairs in holdout_point_pairs.items()
        }

        aggregate[task] = evaluate(
            model, make_examples_from_pairs(world, all_eval_pairs),
            device, batch_size,
        )

        # Predictions between pairs of recovered holdout points
        # themselves - both endpoints were only ever fit against distance
        # probes to normally-trained points, never against each other.
        holdout_to_holdout[task] = (
            evaluate(
                model, make_examples_from_pairs(world, holdout_pairs),
                device, batch_size,
            )
            if holdout_pairs is not None else None
        )

    return {
        "n_holdout_points": len(holdout_indices),
        "n_probes": len(next(iter(holdout_point_pairs.values()))["probe_pairs"]),
        "probe_steps": probe_steps,
        "eval_tasks": list(eval_tasks),
        "per_point": per_point,
        "aggregate": aggregate,
        "holdout_to_holdout": holdout_to_holdout,
    }
