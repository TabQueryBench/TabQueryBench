"""
Vectorized drop-in for `arfpy.arf.arf` (arfpy 0.1.1), imported inside the ARF
container from /work/src/models/arf/adapter (no image change).

Same algorithm and outputs as upstream (src/models/arf/upstream/arf.py); only the
pure-Python / per-group pandas hot loops are replaced by numpy equivalents:

1. Adversarial-loop resampling. Upstream draws n (obs, tree) pairs uniformly,
   groups them by (tree, leaf) and per group samples every column independently
   with replacement from the real rows in that leaf via one `groupby.get_group`
   call per leaf. Here: sort all (tree, leaf) memberships once, then for every
   draw and column pick a uniform random member of the drawn leaf (same
   distribution).
2. Leaf pruning. Upstream calls `np.where(left == tp)` for every pruned node
   (O(#pruned x #nodes) per tree). Here: bottom-up "pruned" flags (a leaf with
   < min_node_size real rows, or an internal node whose two children are both
   pruned) and one vectorized pointer rewrite. The reachable tree is identical;
   trees whose root ends up pruned use the exact upstream loop.
3. FORDE (truncnorm, oob=False, alpha=0). Upstream: per tree a Python loop over
   all nodes for bounds plus pandas melt/merge/groupby. Here: level-wise bounds
   propagation and `np.bincount` statistics. Produces the same `bnds`, `params`
   and `class_probs` tables (row order may differ). Other settings fall back to
   upstream `forde`.
4. FORGE. Upstream merges sampled leaves with all class probabilities and calls
   `groupby("obs").sample(weights=...)` (one Python call per row). Here: per
   variable a group-offset cumulative-probability table and one `searchsorted`.
   Continuous columns use the same truncated normal, vectorized; zero/undefined
   sd returns the leaf mean (upstream intent), and leaf-coverage probabilities
   are renormalized (upstream `np.random.choice` raises when zero-coverage leaves
   were dropped).

RandomForest kwargs (e.g. n_jobs) pass through to sklearn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.ensemble import RandomForestClassifier

from arfpy import arf as _arf_mod


def _leaf_resample(x_real: pd.DataFrame, node_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n, n_trees = node_ids.shape
    m = int(node_ids.max()) + 1
    keys = (np.arange(n_trees, dtype=np.int64)[None, :] * m + node_ids.astype(np.int64)).ravel()
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    obs_of_sorted = order // n_trees

    draw_obs = rng.integers(0, n, size=n)
    draw_tree = rng.integers(0, n_trees, size=n)
    draw_keys = draw_tree.astype(np.int64) * m + node_ids[draw_obs, draw_tree].astype(np.int64)
    start = np.searchsorted(sorted_keys, draw_keys, side="left")
    count = np.searchsorted(sorted_keys, draw_keys, side="right") - start

    out = {}
    for col in x_real.columns:
        pick = start + np.floor(rng.random(n) * count).astype(np.int64)
        pick = np.minimum(pick, start + count - 1)
        out[col] = x_real[col].to_numpy()[obs_of_sorted[pick]]
    return pd.DataFrame(out, columns=x_real.columns)


def _node_depth_levels(left: np.ndarray, right: np.ndarray):
    """Nodes reachable from the root, grouped by depth (list of arrays)."""
    levels = []
    frontier = np.array([0], dtype=np.int64)
    while frontier.size:
        levels.append(frontier)
        internal = frontier[left[frontier] >= 0]
        if not internal.size:
            break
        frontier = np.unique(np.concatenate([left[internal], right[internal]]))
    return levels


def prune_tree_exact(left: np.ndarray, right: np.ndarray, leaf_of_obs: np.ndarray, min_node_size: int) -> None:
    """Upstream arfpy pruning loop, verbatim (in-place on the tree arrays)."""
    leaves = np.where(left < 0)[0]
    unique, counts = np.unique(leaf_of_obs, return_counts=True)
    to_prune = unique[counts < min_node_size]
    to_prune = np.concatenate([to_prune, np.setdiff1d(leaves, unique)])
    while len(to_prune) > 0:
        for tp in to_prune:
            parent = np.where(left == tp)[0]
            if len(parent) > 0:
                left[parent] = right[parent]
            else:
                parent = np.where(right == tp)[0]
                right[parent] = left[parent]
        to_prune = np.where(np.in1d(left, to_prune))[0]


def prune_tree_fast(left: np.ndarray, right: np.ndarray, leaf_of_obs: np.ndarray, min_node_size: int) -> None:
    """Vectorized equivalent of `prune_tree_exact` for the reachable tree."""
    num_nodes = left.shape[0]
    is_leaf = left < 0
    counts = np.bincount(leaf_of_obs, minlength=num_nodes)
    pruned = np.zeros(num_nodes, dtype=bool)
    pruned[is_leaf] = counts[is_leaf] < min_node_size
    left0, right0 = left.copy(), right.copy()
    for lvl in reversed(_node_depth_levels(left0, right0)):
        internal = lvl[left0[lvl] >= 0]
        if internal.size:
            pruned[internal] = pruned[left0[internal]] & pruned[right0[internal]]
    if pruned[0]:
        # Whole tree pruned: pointer choice among pruned children is order dependent
        # and reachable -> replicate upstream exactly.
        prune_tree_exact(left, right, leaf_of_obs, min_node_size)
        return
    internal = np.where(left0 >= 0)[0]
    lp = pruned[left0[internal]]
    rp = pruned[right0[internal]]
    only_l = internal[lp & ~rp]
    only_r = internal[rp & ~lp]
    both = internal[lp & rp]  # unreachable (their parent skips them)
    left[only_l] = right0[only_l]
    right[only_r] = left0[only_r]
    left[both] = right0[both]


class FastARF(_arf_mod.arf):
    def __init__(self, x, num_trees=30, delta=0, max_iters=10, early_stop=True, verbose=True,
                 min_node_size=5, seed=None, prune="fast", **kwargs):
        assert isinstance(x, pd.core.frame.DataFrame), f"expected pandas DataFrame as input, got:{type(x)}"
        assert len(set(list(x))) == x.shape[1], "every column must have a unique column name"
        assert max_iters >= 0, "parameter max_iters must be >= 0"
        assert min_node_size > 0, "parameter min_node_size must be greater than zero"
        assert num_trees > 0, "parameter num_trees must be greater than zero"
        assert 0 <= delta <= 0.5, "parameter delta must be in range 0 <= delta <= 0.5"

        rng = np.random.default_rng(seed)
        x_real = x.copy()
        self.p = x_real.shape[1]
        self.orig_colnames = list(x_real)
        self.num_trees = num_trees
        self.min_node_size = min_node_size

        self.object_cols = x_real.dtypes == "object"
        for col in list(x_real):
            if self.object_cols[col]:
                x_real[col] = x_real[col].astype("category")
        self.factor_cols = x_real.dtypes == "category"
        self.levels = {}
        for col in list(x_real):
            if self.factor_cols[col]:
                self.levels[col] = x_real[col].cat.categories
        for col in list(x_real):
            if self.factor_cols[col]:
                x_real[col] = x_real[col].cat.codes
        self.x_real = x_real

        # initial synthetic data: independent marginals (upstream: per-column permutation)
        x_synth = pd.DataFrame({c: rng.permutation(x_real[c].to_numpy()) for c in x_real.columns}, columns=x_real.columns)
        y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])

        def _fit(xs):
            clf = RandomForestClassifier(oob_score=True, n_estimators=self.num_trees,
                                         min_samples_leaf=min_node_size, **kwargs)
            clf.fit(pd.concat([x_real, xs], ignore_index=True), y)
            return clf

        clf_0 = _fit(x_synth)
        iters = 0
        acc = [clf_0.oob_score_]
        if verbose:
            print(f"Initial accuracy is {acc[0]}", flush=True)
        if acc[0] > 0.5 + delta and iters < max_iters:
            converged = False
            while not converged:
                node_ids = clf_0.apply(x_real)
                x_synth = _leaf_resample(x_real, node_ids, rng)
                del node_ids
                clf_1 = _fit(x_synth)
                acc_1 = clf_1.oob_score_
                acc.append(acc_1)
                iters += 1
                plateau = bool(early_stop and acc[iters] > acc[iters - 1])
                if verbose:
                    print(f"Iteration number {iters} reached accuracy of {acc_1}.", flush=True)
                if acc_1 <= 0.5 + delta or iters >= max_iters or plateau:
                    converged = True
                else:
                    clf_0 = clf_1
        self.clf = clf_0
        self.acc = acc

        if prune != "none":
            self.prune(method=prune)

    def prune(self, method="fast"):
        pruner = prune_tree_fast if method == "fast" else prune_tree_exact
        pred = self.clf.apply(self.x_real)
        for tree_num in range(self.num_trees):
            tree = self.clf.estimators_[tree_num].tree_
            # children_left/right are views into sklearn's node array -> in-place edits stick
            pruner(tree.children_left, tree.children_right, pred[:, tree_num], self.min_node_size)

    def forde(self, dist="truncnorm", oob=False, alpha=0):
        if dist != "truncnorm" or oob or alpha != 0:
            return super().forde(dist=dist, oob=oob, alpha=alpha)
        self.dist, self.oob, self.alpha = dist, oob, alpha

        pred = self.clf.apply(self.x_real)
        n = pred.shape[0]
        names = np.asarray(self.orig_colnames, dtype=object)
        factor = self.factor_cols.to_numpy(dtype=bool)
        cont_j = np.where(~factor)[0]
        fac_j = np.where(factor)[0]
        any_cont = cont_j.size > 0
        xr = {j: self.x_real.iloc[:, j].to_numpy() for j in range(self.p)}

        bnds_parts, params_parts, cp_parts = [], [], []
        f_offset = 0
        for t in range(self.num_trees):
            tree = self.clf.estimators_[t].tree_
            left, right = tree.children_left, tree.children_right
            feat, thr = tree.feature, tree.threshold
            all_leaves = np.where(left < 0)[0]

            # bounds (upstream utils.bnd_fun) on the reachable tree
            lb = np.full((tree.node_count, self.p), -np.inf)
            ub = np.full((tree.node_count, self.p), np.inf)
            frontier = np.array([0], dtype=np.int64)
            while frontier.size:
                internal = frontier[left[frontier] > -1]
                if not internal.size:
                    break
                L, R = left[internal], right[internal]
                lb[L] = lb[internal]
                ub[L] = ub[internal]
                lb[R] = lb[internal]
                ub[R] = ub[internal]
                split = L != R
                fs, ths = feat[internal[split]], thr[internal[split]]
                ub[L[split], fs] = ths
                lb[R[split], fs] = ths
                frontier = np.unique(np.concatenate([L, R]))

            leaves, inv, freq = np.unique(pred[:, t], return_inverse=True, return_counts=True)
            inv = inv.reshape(-1)
            cvg = freq / n
            if any_cont:
                cvg = np.where(cvg == 1 / n, 0.0, cvg)
            keep = cvg > 0
            f_idx = f_offset + np.searchsorted(all_leaves, leaves)
            f_offset += all_leaves.size

            kl = leaves[keep]
            nk = kl.size
            bnds_parts.append(pd.DataFrame({
                "nodeid": np.repeat(kl, self.p),
                "cvg": np.repeat(cvg[keep], self.p),
                "tree": t,
                "variable": np.tile(names, nk),
                "min": lb[kl].ravel(),
                "max": ub[kl].ravel(),
                "f_idx": np.repeat(f_idx[keep], self.p),
            }))

            nl = leaves.size
            min_leaf = np.full((nl, self.p), np.nan)
            max_leaf = np.full((nl, self.p), np.nan)
            min_leaf[keep] = lb[kl]
            max_leaf[keep] = ub[kl]

            for j in cont_j:
                v = xr[j].astype(float)
                ok = ~np.isnan(v)
                cnt = np.bincount(inv[ok], minlength=nl).astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean = np.bincount(inv[ok], weights=v[ok], minlength=nl) / cnt
                    dev = v[ok] - mean[inv[ok]]
                    var = np.bincount(inv[ok], weights=dev * dev, minlength=nl) / (cnt - 1)
                sd = np.where(cnt > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)
                mean = np.where(cnt > 0, mean, np.nan)
                params_parts.append(pd.DataFrame({
                    "tree": t, "nodeid": leaves, "variable": names[j],
                    "mean": mean, "sd": sd, "min": min_leaf[:, j], "max": max_leaf[:, j],
                }))

            for j in fac_j:
                codes = xr[j].astype(np.int64)
                base = int(codes.max()) + 2
                key = inv.astype(np.int64) * base + (codes + 1)
                ukey, kcnt = np.unique(key, return_counts=True)
                li = ukey // base
                sel = keep[li]
                li, kcnt, val = li[sel], kcnt[sel], (ukey[sel] % base) - 1
                cp_parts.append(pd.DataFrame({
                    "f_idx": f_idx[li], "tree": t, "nodeid": leaves[li], "variable": names[j],
                    "value": val, "prob": kcnt / freq[li],
                }))

        self.bnds = pd.concat(bnds_parts, ignore_index=True)
        self.params = pd.concat(params_parts, ignore_index=True) if params_parts else pd.DataFrame()
        self.class_probs = pd.concat(cp_parts, ignore_index=True) if cp_parts else pd.DataFrame()
        return {"cnt": self.params, "cat": self.class_probs, "forest": self.clf,
                "meta": pd.DataFrame(data={"variable": self.orig_colnames, "family": self.dist})}

    def forge(self, n, seed=None):
        if not hasattr(self, "bnds"):
            raise AttributeError("need density estimates to generate data -- run .forde() first!")
        rng = np.random.default_rng(seed)
        n = int(n)
        unique_bnds = self.bnds[["tree", "nodeid", "cvg"]].drop_duplicates()
        p = unique_bnds["cvg"].to_numpy(dtype=float)
        p = p / p.sum()
        draws = rng.choice(unique_bnds.shape[0], size=n, p=p)
        tree_d = unique_bnds["tree"].to_numpy(dtype=np.int64)[draws]
        node_d = unique_bnds["nodeid"].to_numpy(dtype=np.int64)[draws]
        m = int(self.bnds["nodeid"].max()) + 1
        draw_keys = tree_d * m + node_d

        data_new = {}
        if np.invert(self.factor_cols).any():
            params = self.params
        if self.factor_cols.any():
            class_probs = self.class_probs

        for j in range(self.p):
            colname = self.orig_colnames[j]
            if self.factor_cols.iloc[j]:
                cp = class_probs[class_probs["variable"] == colname]
                keys = cp["tree"].to_numpy(dtype=np.int64) * m + cp["nodeid"].to_numpy(dtype=np.int64)
                order = np.argsort(keys, kind="stable")
                keys = keys[order]
                probs = cp["prob"].to_numpy(dtype=float)[order]
                values = cp["value"].to_numpy()[order]
                ukeys, gstart, gcount = np.unique(keys, return_index=True, return_counts=True)
                gid = np.repeat(np.arange(len(ukeys)), gcount)
                csum = np.cumsum(probs)
                offset = np.where(gstart > 0, csum[np.maximum(gstart - 1, 0)], 0.0)
                within = csum - offset[gid]
                total = within[gstart + gcount - 1]
                total = np.where(total > 0, total, 1.0)
                glob = gid + within / total[gid]
                g = np.searchsorted(ukeys, draw_keys)
                if np.any(g >= len(ukeys)) or np.any(ukeys[np.minimum(g, len(ukeys) - 1)] != draw_keys):
                    raise RuntimeError(f"FastARF.forge: missing class probabilities for '{colname}'")
                pos = np.searchsorted(glob, g + rng.random(n), side="right")
                pos = np.clip(pos, gstart[g], gstart[g] + gcount[g] - 1)
                data_new[colname] = values[pos]
            else:
                pr = params[params["variable"] == colname]
                keys = pr["tree"].to_numpy(dtype=np.int64) * m + pr["nodeid"].to_numpy(dtype=np.int64)
                order = np.argsort(keys, kind="stable")
                keys = keys[order]
                g = np.searchsorted(keys, draw_keys)
                if np.any(g >= len(keys)) or np.any(keys[np.minimum(g, len(keys) - 1)] != draw_keys):
                    raise RuntimeError(f"FastARF.forge: missing leaf parameters for '{colname}'")
                idx = order[g]
                loc = pr["mean"].to_numpy(dtype=float)[idx]
                scale = pr["sd"].to_numpy(dtype=float)[idx]
                lo = pr["min"].to_numpy(dtype=float)[idx]
                hi = pr["max"].to_numpy(dtype=float)[idx]
                out = loc.copy()
                ok = np.isfinite(scale) & (scale > 0)
                if ok.any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        a = (lo[ok] - loc[ok]) / scale[ok]
                        b = (hi[ok] - loc[ok]) / scale[ok]
                        draws_c = scipy.stats.truncnorm(a=a, b=b, loc=loc[ok], scale=scale[ok]).rvs(
                            size=int(ok.sum()), random_state=rng
                        )
                    out[ok] = draws_c
                bad = ~np.isfinite(out)
                if bad.any():
                    out[bad] = np.clip(loc[bad], lo[bad], hi[bad])
                data_new[colname] = out

        df = pd.DataFrame(data_new, columns=self.orig_colnames)
        for col in self.orig_colnames:
            if self.factor_cols[col]:
                df[col] = pd.Categorical.from_codes(df[col].astype(int), categories=self.levels[col])
        for col in self.orig_colnames:
            if self.object_cols[col]:
                df[col] = df[col].astype("object")
        return df
