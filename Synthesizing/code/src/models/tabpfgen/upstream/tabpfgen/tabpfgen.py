import hashlib
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

_TABPFGEN_BACKEND = os.environ.get("TABPFGEN_BACKEND", "local").strip().lower()
if _TABPFGEN_BACKEND == "client":
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor
    from tabpfn_client import constants as _client_constants
    from tabpfn_client.service_wrapper import UserAuthenticationClient as _UserAuthenticationClient

    _cache_root = Path(
        os.environ.get("TABPFN_CLIENT_CACHE_DIR")
        or os.environ.get("XDG_CACHE_HOME")
        or "/tmp"
    ) / "tabpfn_client"
    _client_constants.CACHE_DIR = _cache_root
    _UserAuthenticationClient.CACHED_TOKEN_FILE = _cache_root / "config"

    _token = os.environ.get("TABPFN_TOKEN", "").strip()
    if _token:
        tabpfn_client.set_access_token(_token)
else:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.preprocessing._encoders"
)

# Default checkpoint filenames per TabPFN model version (mirrors tabpfn 6.4.1
# `tabpfn.model_loading.ModelSource.*.default_filename`).
MODEL_DEFAULT_FILENAMES = {
    "v2": {
        "classifier": "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
        "regressor": "tabpfn-v2-regressor.ckpt",
    },
    "v2.5": {
        "classifier": "tabpfn-v2.5-classifier-v2.5_default.ckpt",
        "regressor": "tabpfn-v2.5-regressor-v2.5_default.ckpt",
    },
}
# TabPFN (v2 and v2.5) hard limit; not ignorable.
MAX_TABPFN_CLASSES = 10


class TabPFGen:
    def __init__(
        self,
        n_sgld_steps: int = 1000,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        device: str = "auto",
        *,
        model_version: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        regressor_model_path: Optional[str] = None,
        n_estimators: Optional[int] = None,
        fit_mode: Optional[str] = None,
        ignore_pretraining_limits: bool = False,
        many_class: str = "auto",
        label_sampling: str = "argmax",
        regression_sampling: str = "upstream",
        progress_every: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize TabPFGen with SGLD parameters.

        Args:
            n_sgld_steps: Number of SGLD steps to take (Default: 1000)
            sgld_step_size: Step size for SGLD updates (Default: 0.01)
            sgld_noise_scale: Noise scale for SGLD updates (Default: 0.01)
            device: "auto" -> cuda if available else cpu.
            model_version: "v2" | "v2.5" | None (None = tabpfn library default).
            classifier_model_path / regressor_model_path: explicit checkpoint files.
            n_estimators: TabPFN ensemble size (None = library default).
            fit_mode: TabPFN fit_mode (None = library default).
            ignore_pretraining_limits: forwarded to TabPFN.
            many_class: "auto" wraps the classifier in tabpfn_extensions
                ManyClassClassifier when >10 classes; "off" raises instead.
            label_sampling: "argmax" (upstream) or "sample" (draw from predict_proba).
            regression_sampling: "upstream" (quantile heuristic), "predictive"
                (sample the TabPFN bar distribution), or "median".
            progress_every: print SGLD progress every N steps (0 = silent).
            seed: random_state forwarded to TabPFN.
        """
        if model_version is not None and model_version not in MODEL_DEFAULT_FILENAMES:
            raise ValueError(
                f"Unsupported TabPFN model_version={model_version!r}; "
                f"expected one of {sorted(MODEL_DEFAULT_FILENAMES)}"
            )
        if label_sampling not in ("argmax", "sample"):
            raise ValueError(f"Invalid label_sampling={label_sampling!r}")
        if regression_sampling not in ("upstream", "predictive", "median"):
            raise ValueError(f"Invalid regression_sampling={regression_sampling!r}")
        if many_class not in ("auto", "off"):
            raise ValueError(f"Invalid many_class={many_class!r}")
        self.n_sgld_steps = n_sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise_scale = sgld_noise_scale
        self.scaler = StandardScaler()
        self.device = self._infer_device(device)
        self.model_version = model_version
        self.classifier_model_path = classifier_model_path
        self.regressor_model_path = regressor_model_path
        self.n_estimators = n_estimators
        self.fit_mode = fit_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.many_class = many_class
        self.label_sampling = label_sampling
        self.regression_sampling = regression_sampling
        self.progress_every = int(progress_every or 0)
        self.seed = seed
        self._fit_cache_key = None
        self._fit_cache_model = None

    # ------------------------------------------------------------------ TabPFN

    def _tabpfn_kwargs(self, kind: str) -> dict:
        kwargs = {"device": self.device}
        path = (
            self.classifier_model_path
            if kind == "classifier"
            else self.regressor_model_path
        )
        if path is None and self.model_version is not None:
            from tabpfn.model_loading import prepend_cache_path

            path = prepend_cache_path(MODEL_DEFAULT_FILENAMES[self.model_version][kind])
        if path is not None:
            kwargs["model_path"] = str(path)
        if self.n_estimators:
            kwargs["n_estimators"] = int(self.n_estimators)
        if self.fit_mode:
            kwargs["fit_mode"] = self.fit_mode
        if self.ignore_pretraining_limits:
            kwargs["ignore_pretraining_limits"] = True
        if self.seed is not None:
            kwargs["random_state"] = int(self.seed)
        return kwargs

    def _make_classifier(self, n_classes: Optional[int] = None):
        if _TABPFGEN_BACKEND == "client":
            return TabPFNClassifier()
        clf = TabPFNClassifier(**self._tabpfn_kwargs("classifier"))
        if n_classes is not None and n_classes > MAX_TABPFN_CLASSES:
            if self.many_class == "off":
                raise ValueError(
                    f"Target has {n_classes} classes but TabPFN supports at most "
                    f"{MAX_TABPFN_CLASSES}; set TABPFGEN_MANY_CLASS=auto to use "
                    "tabpfn_extensions.many_class.ManyClassClassifier."
                )
            try:
                from tabpfn_extensions.many_class import ManyClassClassifier
            except ImportError as exc:  # pragma: no cover - image dependent
                raise RuntimeError(
                    f"Target has {n_classes} classes (> {MAX_TABPFN_CLASSES}) and "
                    "tabpfn_extensions is not installed; cannot wrap TabPFN."
                ) from exc
            print(
                f"[TabPFGen] {n_classes} classes > {MAX_TABPFN_CLASSES}: "
                "using ManyClassClassifier(alphabet_size=10)"
            )
            clf = ManyClassClassifier(
                estimator=clf,
                alphabet_size=MAX_TABPFN_CLASSES,
                random_state=self.seed,
            )
        return clf

    def _make_regressor(self):
        if _TABPFGEN_BACKEND == "client":
            return TabPFNRegressor()
        return TabPFNRegressor(**self._tabpfn_kwargs("regressor"))

    @staticmethod
    def _fingerprint(*arrays: np.ndarray) -> str:
        h = hashlib.blake2b(digest_size=16)
        for a in arrays:
            a = np.ascontiguousarray(a)
            h.update(repr((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def _fitted(self, kind: str, X_scaled: np.ndarray, y: np.ndarray, n_classes=None):
        """Fit the TabPFN label model once per (kind, data) and reuse it across chunks."""
        key = (kind, self._fingerprint(X_scaled, y))
        if self._fit_cache_key != key:
            t0 = time.perf_counter()
            model = (
                self._make_classifier(n_classes)
                if kind == "classifier"
                else self._make_regressor()
            )
            model.fit(X_scaled, y)
            self._fit_cache_key = key
            self._fit_cache_model = model
            print(
                f"[TabPFGen] fitted TabPFN {kind} on {len(X_scaled)} rows "
                f"in {time.perf_counter() - t0:.1f}s"
            )
        return self._fit_cache_model

    def prepare(self, X_train: np.ndarray, y_train: np.ndarray, task: str):
        """Load weights and fit the label model up front (fail fast before SGLD)."""
        X_scaled = StandardScaler().fit_transform(X_train)
        if task == "classification":
            y = np.asarray(y_train)
            return self._fitted("classifier", X_scaled, y, n_classes=len(np.unique(y)))
        y = np.asarray(y_train, dtype=np.float64)
        y_mean, y_std = self._y_stats(y)
        return self._fitted("regressor", X_scaled, (y - y_mean) / y_std)

    @staticmethod
    def _y_stats(y: np.ndarray) -> Tuple[float, float]:
        y_mean, y_std = float(np.mean(y)), float(np.std(y))
        if not np.isfinite(y_std) or y_std <= 0:
            y_std = 1.0
        return y_mean, y_std

    # -------------------------------------------------------------------- SGLD

    def _infer_device(self, device: str | torch.device | None) -> torch.device:
        """
        Infer the device and data type from the given device string.

        Args:
            device: The device to infer the type from.

        Returns:
            The inferred device
        """
        if (device is None) or (isinstance(device, str) and device == "auto"):
            device_type_ = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(device_type_)
        if isinstance(device, str):
            return torch.device(device)
        if isinstance(device, torch.device):
            return device
        raise ValueError(f"Invalid device: {device}")

    def _compute_energy(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute differentiable energy score using surrogate network
        """
        # Use difference from training samples as a proxy for energy
        distances = torch.cdist(x_synth, x_train)
        min_distances, _ = distances.min(dim=1)

        # Add class-conditional term
        class_mask = y_synth.unsqueeze(1) == y_train.unsqueeze(0)
        class_distances = distances * class_mask.float()
        class_distances = class_distances.sum(dim=1) / (
            class_mask.float().sum(dim=1) + 1e-6
        )

        # Combine terms
        energy = min_distances + class_distances
        return energy

    def _sgld_step(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one SGLD step
        """
        x_synth = x_synth.clone().detach().requires_grad_(True)

        # Compute energy and its gradient
        energy = self._compute_energy(x_synth, y_synth, x_train, y_train)
        energy_sum = energy.sum()

        # Compute gradients with allow_unused=True
        grad = torch.autograd.grad(
            energy_sum,
            x_synth,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]

        if grad is None:
            grad = torch.zeros_like(x_synth)

        # Update using gradients and noise
        noise = torch.randn_like(x_synth) * np.sqrt(2 * self.sgld_step_size)
        x_synth_new = (
            x_synth - self.sgld_step_size * grad + self.sgld_noise_scale * noise
        )

        return x_synth_new.detach()

    def _run_sgld(self, x_synth, y_synth, x_train, y_train, label: str = ""):
        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train, y_train)
            if self.progress_every and step % self.progress_every == 0:
                print(f"{label}Step {step}/{self.n_sgld_steps}")
        return x_synth

    def _generate_samples_for_class(
        self,
        class_label: int,
        n_samples: int,
        x_train_scaled: torch.Tensor,
        y_train: torch.Tensor,
        X_train_scaled: np.ndarray,
    ) -> torch.Tensor:
        """
        Generate synthetic samples for a specific class using SGLD.
        """
        class_indices = torch.where(y_train == class_label)[0]
        sample_indices = torch.randint(0, len(class_indices), (n_samples,))
        selected_indices = class_indices[sample_indices]
        x_synth = (
            x_train_scaled[selected_indices]
            + torch.randn(n_samples, X_train_scaled.shape[1], device=self.device) * 0.01
        )
        y_synth = torch.full((n_samples,), class_label, device=self.device)
        return self._run_sgld(
            x_synth, y_synth, x_train_scaled, y_train, label=f"  Class {class_label}: "
        )

    def balance_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_per_class: Optional[int] = None,
        min_class_size: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Balance dataset by generating synthetic samples for underrepresented classes.
        (Upstream helper; not used by the benchmark bridge.)
        """
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples")

        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        max_class_size = max(class_counts)

        if target_per_class is None:
            target_size = max_class_size
        else:
            if target_per_class < max_class_size:
                raise ValueError(
                    f"target_per_class ({target_per_class}) must be >= largest class size ({max_class_size})"
                )
            target_size = target_per_class

        valid_classes = []
        synthetic_needed = {}
        for cls, count in class_distribution.items():
            if count < min_class_size:
                print(
                    f"Warning: Skipping class {cls} (only {count} samples, below threshold {min_class_size})"
                )
            else:
                valid_classes.append(cls)
                synthetic_needed[cls] = max(0, target_size - count)

        if not valid_classes:
            raise ValueError("No classes meet the minimum size requirement")

        total_synthetic = sum(synthetic_needed.values())
        if total_synthetic == 0:
            return (
                np.array([]).reshape(0, X_train.shape[1]),
                np.array([]),
                X_train.copy(),
                y_train.copy(),
            )

        X_scaled = self.scaler.fit_transform(X_train)
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, device=self.device)

        x_synthetic_list = []
        for cls in valid_classes:
            n_needed = synthetic_needed[cls]
            if n_needed > 0:
                x_synthetic_list.append(
                    self._generate_samples_for_class(
                        cls, n_needed, x_train, y_train_tensor, X_scaled
                    )
                )

        x_synthetic_np = torch.cat(x_synthetic_list, dim=0).detach().cpu().numpy()
        valid_mask = np.isin(y_train, valid_classes)
        X_train_valid = X_scaled[valid_mask]
        y_train_valid = y_train[valid_mask]
        clf = self._make_classifier(len(np.unique(y_train_valid)))
        clf.fit(X_train_valid, y_train_valid)
        probs = clf.predict_proba(x_synthetic_np)
        y_synthetic_mapped = np.asarray(clf.classes_)[probs.argmax(axis=1)]
        X_synthetic = self.scaler.inverse_transform(x_synthetic_np)

        X_combined = np.vstack([X_train, X_synthetic])
        y_combined = np.concatenate([y_train, y_synthetic_mapped])
        return X_synthetic, y_synthetic_mapped, X_combined, y_combined

    # -------------------------------------------------------------- generation

    def generate_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        balance_classes: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for classification.

        Args:
            X_train: shape (n_samples, n_features)
            y_train: shape (n_samples,)
            n_samples: number of synthetic samples to generate (returned exactly)
            balance_classes: if True, initialise SGLD with equal counts per class
                (upstream default). Default False: initialise from training rows
                drawn uniformly, which preserves the empirical class distribution.

        Returns:
            Tuple of synthetic features and labels (labels are values of y_train).
        """
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        X_scaled = self.scaler.fit_transform(X_train)
        y_np = np.asarray(y_train)
        classes = np.unique(y_np)

        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train_t = torch.as_tensor(y_np, device=self.device)

        if balance_classes:
            base, rem = divmod(n_samples, len(classes))
            per_class = np.full(len(classes), base, dtype=int)
            if rem:
                per_class[np.random.choice(len(classes), size=rem, replace=False)] += 1
            init_idx = np.concatenate(
                [
                    np.random.choice(np.where(y_np == cls)[0], size=k)
                    for cls, k in zip(classes, per_class)
                    if k > 0
                ]
            )
        else:
            init_idx = np.random.choice(len(y_np), size=n_samples, replace=True)

        init_idx_t = torch.as_tensor(init_idx, device=self.device)
        x_synth = (
            x_train[init_idx_t]
            + torch.randn(n_samples, X_scaled.shape[1], device=self.device) * 0.01
        )
        y_synth = y_train_t[init_idx_t]

        x_synth = self._run_sgld(x_synth, y_synth, x_train, y_train_t)

        # Refine labels using TabPFN (fitted once, reused across calls on same data)
        x_synth_np = x_synth.detach().cpu().numpy()
        clf = self._fitted("classifier", X_scaled, y_np, n_classes=len(classes))
        probs = np.asarray(clf.predict_proba(x_synth_np))
        model_classes = np.asarray(clf.classes_)
        if self.label_sampling == "sample":
            cum = np.cumsum(probs, axis=1)
            u = np.random.rand(len(probs), 1) * cum[:, -1:]
            idx = np.clip((cum < u).sum(axis=1), 0, probs.shape[1] - 1)
        else:
            idx = probs.argmax(axis=1)
        y_out = model_classes[idx]

        X_synth = self.scaler.inverse_transform(x_synth_np)
        return X_synth, y_out

    def generate_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        use_quantiles: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for regression.

        Errors from TabPFN are propagated (no silent random-y fallback).
        """
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        X_scaled = self.scaler.fit_transform(X_train)
        y_train = np.asarray(y_train, dtype=np.float64)
        y_mean, y_std = self._y_stats(y_train)
        y_scaled = (y_train - y_mean) / y_std

        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)

        # Stratified initialisation over (disjoint) target-quantile strata,
        # allocating exactly n_samples proportionally to stratum sizes.
        n_strata = 10
        inner_edges = np.unique(np.quantile(y_train, np.linspace(0, 1, n_strata + 1)[1:-1]))
        strata = np.searchsorted(inner_edges, y_train, side="right")
        ids, counts = np.unique(strata, return_counts=True)
        exact = counts * n_samples / counts.sum()
        alloc = np.floor(exact).astype(int)
        short = n_samples - int(alloc.sum())
        if short > 0:
            alloc[np.argsort(-(exact - alloc))[:short]] += 1

        x_synth_list = []
        for sid, k in zip(ids, alloc):
            if k <= 0:
                continue
            members = np.where(strata == sid)[0]
            sampled = np.random.choice(members, size=int(k))
            stratum_std = np.std(X_scaled[members], axis=0)
            noise = np.random.normal(0, stratum_std * 0.1, (int(k), X_scaled.shape[1]))
            x_synth_list.append(X_scaled[sampled] + noise)
        x_synth = torch.tensor(
            np.vstack(x_synth_list), device=self.device, dtype=torch.float32
        )

        # SGLD (upstream computed an "adaptive" step size that was never applied;
        # the fixed sgld_step_size is used, as in upstream behaviour).
        zeros_synth = torch.zeros(len(x_synth), device=self.device)
        zeros_train = torch.zeros(len(x_train), device=self.device)
        x_synth = self._run_sgld(x_synth, zeros_synth, x_train, zeros_train)

        x_synth_np = x_synth.detach().cpu().numpy()
        regressor = self._fitted("regressor", X_scaled, y_scaled)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = regressor.predict(x_synth_np, output_type="full")

        mode = self.regression_sampling if use_quantiles else "median"
        if mode == "predictive":
            criterion = predictions["criterion"]
            logits = predictions["logits"]
            with torch.no_grad():
                y_synth = criterion.sample(logits.float()).detach().cpu().numpy()
            y_synth = np.asarray(y_synth, dtype=np.float64).reshape(-1)
        elif mode == "upstream":
            quantiles = predictions["quantiles"]
            if not isinstance(quantiles, list):
                quantiles = [quantiles]
            n_quantiles = len(quantiles)
            probs = np.abs(np.random.normal(0, 0.5, size=len(x_synth_np)))
            quantile_idx = (probs * n_quantiles).astype(int).clip(0, n_quantiles - 1)
            y_synth = np.array([quantiles[i][j] for j, i in enumerate(quantile_idx)])
            y_synth = y_synth + np.random.normal(0, 0.01, size=len(y_synth))
        else:
            y_synth = np.asarray(predictions["median"], dtype=np.float64)
            y_synth = y_synth + np.random.normal(0, 0.01, size=len(y_synth))

        if len(y_synth) != len(x_synth_np) or not np.all(np.isfinite(y_synth)):
            raise RuntimeError(
                f"TabPFN regression produced invalid targets "
                f"(len={len(y_synth)}, expected={len(x_synth_np)}, "
                f"non_finite={int(np.sum(~np.isfinite(y_synth)))})"
            )

        X_synth = self.scaler.inverse_transform(x_synth_np)
        y_synth = y_synth * y_std + y_mean
        return X_synth, y_synth
