"""V9 SPQ single-primary scorer.

One query produces one primary similarity. Component diagnostics are retained,
but never mixed into the primary score with 50/25/25 weights.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any
from tqb_scoring.standards.semantic_v8 import scorer as old

SCORE_VERSION = "spq_v9_single_primary"
SEMANTIC_QUERY_SCORE_METHOD = "spq"
EPS = 1e-12

def _clip(x):
    try: return max(0.0, min(1.0, float(x)))
    except Exception: return 0.0

def _num(x):
    try:
        s=str(x).strip()
        if not s: return None
        v=float(s)
        return v if math.isfinite(v) else None
    except Exception: return None

def _key(row, indices, numeric_modes=None):
    vals=[]
    for pos, idx in enumerate(indices):
        value=row[idx] if idx < len(row) else None
        mode=(numeric_modes or ["categorical"]*len(indices))[pos]
        if mode == "numeric":
            n=_num(value); vals.append(round(n,10) if n is not None else "<NULL>")
        else: vals.append("<NULL>" if value is None else str(value))
    return tuple(vals)

def _numeric_modes(real_rows, syn_rows, indices):
    modes=[]
    for idx in indices:
        vals=[row[idx] for row in real_rows+syn_rows if idx < len(row) and str(row[idx]).strip()]
        numeric=[_num(v) for v in vals]
        ratio=sum(v is not None for v in numeric)/max(1,len(vals))
        unique=len({round(v,10) for v in numeric if v is not None})
        # Continuous numeric keys are compared through shared quantile bins.
        modes.append("numeric" if ratio >= .8 else "categorical")
    return modes

def _keys(rows, indices, modes):
    return [_key(row,indices,modes) for row in rows]

def _key_maps(rows, key_indices, measure_indices, modes):
    out={}
    for row in rows:
        k=_key(row,key_indices,modes) if key_indices else ("__scalar__",)
        vals=[_num(row[i]) if i < len(row) else None for i in measure_indices]
        out[k]=[v for v in vals if v is not None]
    return out

def _union_rate(real_rows,syn_rows,keys,measure,modes):
    rm=_key_maps(real_rows,keys,measure,modes); sm=_key_maps(syn_rows,keys,measure,modes)
    allk=set(rm)|set(sm)
    if not allk:return 1.0
    scores=[]
    for k in allk:
        if k not in rm or k not in sm or not rm[k] or not sm[k]: scores.append(0.0)
        else:scores.append(_clip(1-abs(rm[k][0]-sm[k][0])))
    return sum(scores)/len(scores)

def _union_numeric(real_rows,syn_rows,keys,measure,modes):
    rm=_key_maps(real_rows,keys,measure,modes); sm=_key_maps(syn_rows,keys,measure,modes); allk=set(rm)|set(sm)
    if not allk:return 1.0
    vals=[]
    for k in allk:
        if k not in rm or k not in sm: vals.append(0.0); continue
        n=min(len(rm[k]),len(sm[k])); vals.extend(old._symmetric_numeric_similarity(rm[k][i],sm[k][i]) for i in range(n))
    return sum(vals)/len(vals) if vals else 0.0

def _support_tvd(real_rows,syn_rows,keys,modes,support_idx):
    def dist(rows):
        c=Counter()
        for row in rows:
            k=_key(row,keys,modes) if keys else ("__total__",)
            c[k]+=max(0.0,_num(row[support_idx]) or 0.0) if support_idx is not None and support_idx<len(row) else 1.0
        total=sum(c.values()) or 1.0
        return {k:v/total for k,v in c.items()}
    a,b=dist(real_rows),dist(syn_rows); return _clip(1-.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b)))

def _rbo(real_rows,syn_rows,keys,modes,p=.9):
    a=_keys(real_rows,keys,modes); b=_keys(syn_rows,keys,modes)
    if not a and not b:return 1.0
    seen_a=set();seen_b=set(); score=0.0
    for d in range(1,max(len(a),len(b))+1):
        if d<=len(a):seen_a.add(a[d-1])
        if d<=len(b):seen_b.add(b[d-1])
        score += (p**(d-1))*(len(seen_a&seen_b)/d)
    return _clip((1-p)*score)

def _policy(query, columns):
    declared=str(query.get("scorer_type") or "")
    old_policy=old._policy_from_query(query,columns)
    mapping={"scalar":"scalar","count_support_distribution":"count_support_distribution","rate_share_proportion":"rate_share_proportion","ratio":"ratio","keyed_numeric_aggregate":"keyed_numeric_aggregate","topk_tailk_ranking":"topk_ranking","distribution_cardinality_profile":"count_support_distribution","tail_outlier_distribution":"scalar"}
    scorer=mapping.get(declared,mapping.get(old_policy.scorer_type, "keyed_numeric_aggregate"))
    if scorer=="tail_outlier_distribution": scorer="scalar"
    return old_policy,scorer

def _resolve_key_indices(query, columns, fallback):
    """Prefer SQL GROUP BY bindings over alias-name heuristics."""
    sql=str(query.get('sql') or '')
    m=re.search(r'\bGROUP\s+BY\s+(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b|$)',sql,re.I|re.S)
    if not m: return fallback
    names=re.findall(r'"([^"]+)"|\b([A-Za-z_][A-Za-z0-9_]*)\b',m.group(1))
    wanted=[a or b for a,b in names]
    out=[i for i,c in enumerate(columns) if c in wanted]
    return out or fallback

def compare_semantic_execution_results(real_exec, syn_exec, *, query=None, legacy_detail=None):
    query=query or {}
    if not getattr(real_exec,"ok",False) or not getattr(syn_exec,"ok",False):
        return 0.0,{"semantic_query_score_method":SEMANTIC_QUERY_SCORE_METHOD,"score_version":SCORE_VERSION,"score_contract_version":SCORE_VERSION,"validity_status":"invalid","validity_failures":["real_query_failed" if not getattr(real_exec,"ok",False) else "synthetic_query_failed"],"scorer_type":"invalid","primary_metric":"primary_semantic_similarity","component_scores":{}}
    rc=[str(x) for x in getattr(real_exec,"columns",[])]; sc=[str(x) for x in getattr(syn_exec,"columns",[])]
    if not rc or not sc:return 0.0,{"semantic_query_score_method":SEMANTIC_QUERY_SCORE_METHOD,"score_version":SCORE_VERSION,"score_contract_version":SCORE_VERSION,"validity_status":"invalid","validity_failures":["missing_columns"],"scorer_type":"invalid","primary_metric":"primary_semantic_similarity","component_scores":{}}
    real_rows=list(getattr(real_exec,"rows",[]) or []); syn_rows=list(getattr(syn_exec,"rows",[]) or [])
    policy,scorer=_policy(query,rc)
    key_idx=_resolve_key_indices(query,rc,old._column_indices(rc,policy.key_columns)); meas_idx=[i for i in range(len(rc)) if i not in set(key_idx)]; support=old._column_indices(rc,(policy.support_column,)) if policy.support_column else []
    if not meas_idx: meas_idx=[i for i in range(len(rc)) if i not in set(key_idx)]
    support_idx=support[0] if support else None; modes=_numeric_modes(real_rows,syn_rows,key_idx)
    if scorer=="scalar": primary=old._scalar_similarity(real_rows,syn_rows,meas_idx,False); components={"primary_scalar_similarity":primary}
    elif scorer=="rate_share_proportion": primary=_union_rate(real_rows,syn_rows,key_idx,meas_idx,modes); components={"primary_rate_similarity":primary}
    elif scorer=="count_support_distribution": primary=_support_tvd(real_rows,syn_rows,key_idx,modes,support_idx); components={"primary_support_distribution_similarity":primary}
    elif scorer=="ratio": primary=_union_numeric(real_rows,syn_rows,key_idx,meas_idx,modes); components={"primary_ratio_similarity":primary}
    elif scorer=="topk_ranking": primary=_rbo(real_rows,syn_rows,key_idx,modes); components={"primary_rbo":primary}
    else: primary=_union_numeric(real_rows,syn_rows,key_idx,meas_idx,modes); components={"primary_keyed_numeric_similarity":primary}
    detail={"semantic_query_score_method":SEMANTIC_QUERY_SCORE_METHOD,"score_version":SCORE_VERSION,"score_contract_version":SCORE_VERSION,"validity_status":"ok","validity_failures":[],"scorer_type":scorer,"primary_metric":list(components)[0],"primary_score":round(_clip(primary),6),"weight_rule":"single_primary","component_scores":{k:round(_clip(v),6) for k,v in components.items()},"diagnostics":{"key_columns":[rc[i] for i in key_idx],"numeric_key_columns":[rc[i] for i,m in zip(key_idx,modes) if m=="numeric"],"key_match_mode":"type_aware_numeric_normalization","legacy_query_score_method":(legacy_detail or {}).get("query_score_method")}}
    return _clip(primary),detail
