import os, json
import numpy as np

def load(path):
    return np.load(path, allow_pickle=True)

def main(data_dir: str):
    # required files
    req = [
        "X_num_train.npy","X_num_val.npy","X_num_test.npy",
        "X_cat_train.npy","X_cat_val.npy","X_cat_test.npy",
        "y_train.npy","y_val.npy","y_test.npy"
    ]
    for f in req:
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    Xn_tr = load(os.path.join(data_dir,"X_num_train.npy"))
    Xc_tr = load(os.path.join(data_dir,"X_cat_train.npy"))
    y_tr  = load(os.path.join(data_dir,"y_train.npy"))

    # basic dims
    n_num = 0 if Xn_tr.ndim < 2 else int(Xn_tr.shape[1])
    n_cat = 0 if Xc_tr.ndim < 2 else int(Xc_tr.shape[1])

    # infer task / y info
    y_flat = y_tr.reshape(-1)
    uniq = np.unique(y_flat)
    # if y is integer and has few unique values, could be classification
    is_int = np.issubdtype(y_flat.dtype, np.integer)
    num_classes = int(len(uniq)) if is_int else 0
    
    # determine task_type
    if is_int and num_classes == 2:
        task_type = "binclass"
    elif is_int and num_classes > 2 and num_classes <= 100:
        task_type = "multiclass"
    else:
        task_type = "regression"

    # cat sizes (per categorical column)
    cat_sizes = []
    if n_cat > 0:
        # compute max+1 per column (assume categories encoded 0..K-1)
        for j in range(n_cat):
            col = Xc_tr[:, j].reshape(-1)
            if col.size == 0:
                cat_sizes.append(0)
            else:
                mx = int(np.max(col))
                cat_sizes.append(mx + 1)

    # numeric stats (optional but useful)
    num_stats = {}
    if n_num > 0:
        # mean/std/min/max over train numeric
        num_stats = {
            "mean": np.mean(Xn_tr, axis=0).tolist(),
            "std":  (np.std(Xn_tr, axis=0) + 1e-12).tolist(),
            "min":  np.min(Xn_tr, axis=0).tolist(),
            "max":  np.max(Xn_tr, axis=0).tolist(),
        }

    # This repo expects info.json. Keep fields simple & robust.
    info = {
        "task_type": task_type,
        "n_num_features": n_num,
        "n_cat_features": n_cat,
        "cat_sizes": cat_sizes,
        "y_dtype": str(y_flat.dtype),
        "y_unique_count": int(len(uniq)),
        "y_unique_head": uniq[:20].tolist(),
        # heuristics: user can override in config.toml
        "is_classification_like": bool(is_int and len(uniq) <= 100),
        "num_classes_like": num_classes,
    }

    # write files
    with open(os.path.join(data_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # some codepaths may look for these (harmless if unused)
    with open(os.path.join(data_dir, "cat_sizes.json"), "w", encoding="utf-8") as f:
        json.dump({"cat_sizes": cat_sizes}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(data_dir, "num_stats.json"), "w", encoding="utf-8") as f:
        json.dump(num_stats, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", os.path.join(data_dir,"info.json"))
    print("[OK] n_num =", n_num, "n_cat =", n_cat, "cat_sizes =", cat_sizes)
    print("[OK] y unique count =", len(uniq), "head =", uniq[:20])

if __name__ == "__main__":
    data_dir = "data/Tab-Cate-1"
    main(data_dir)

