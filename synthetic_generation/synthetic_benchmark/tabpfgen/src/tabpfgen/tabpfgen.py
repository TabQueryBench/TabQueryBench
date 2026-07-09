import numpy as np
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.preprocessing._encoders"
)


class TabPFGen:
    def __init__(
        self,
        n_sgld_steps: int = 1000,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        device: str = "auto",
    ):
        """
        Initialize TabPFGen with SGLD parameters.

        Args:
            n_sgld_steps: int
                Number of SGLD steps to take (Default: 1000)
            sgld_step_size: float
                Step size for SGLD updates (Default: 0.01)
            sgld_noise_scale: float
                Noise scale for SGLD updates (Default: 0.01)
            device: str, torch.device
                Device to use for computation (Default: "auto"), If `"auto"`, the device is `"cuda"` if available, otherwise `"cpu"`.
        """
        self.n_sgld_steps = n_sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise_scale = sgld_noise_scale
        self.scaler = StandardScaler()
        self.device = self._infer_device(device)

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

        return x_synth_new

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

        Args:
            class_label: The class to generate samples for
            n_samples: Number of samples to generate
            x_train_scaled: Scaled training features as tensor
            y_train: Training labels as tensor
            X_train_scaled: Scaled training features as numpy array

        Returns:
            Generated samples as tensor
        """
        # Get indices for this class
        class_indices = torch.where(y_train == class_label)[0]

        # Initialize synthetic samples near existing class samples
        sample_indices = torch.randint(0, len(class_indices), (n_samples,))
        selected_indices = class_indices[sample_indices]
        x_synth = (
            x_train_scaled[selected_indices]
            + torch.randn(n_samples, X_train_scaled.shape[1], device=self.device) * 0.01
        )
        y_synth = torch.full((n_samples,), class_label, device=self.device)

        # SGLD iterations
        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train_scaled, y_train)

            if step % 200 == 0:
                print(f"  Class {class_label}: Step {step}/{self.n_sgld_steps}")

        return x_synth

    def balance_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_per_class: Optional[int] = None,
        min_class_size: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Balance dataset by generating synthetic samples for underrepresented classes.

        Note:
            The final class distribution may be approximately balanced rather than
            perfectly balanced due to TabPFN's label refinement process, which
            prioritizes data quality over exact class counts. Classes smaller than
            min_class_size are excluded from synthetic generation but remain in
            the combined dataset.

        Args:
            X_train: np.ndarray
                Input features for training, shape (n_samples, n_features)
            y_train: np.ndarray
                Target labels for training, shape (n_samples,)
            target_per_class: Optional[int]
                Target number of samples per class. If None, uses majority class size.
            min_class_size: int
                Minimum class size to include in balancing (Default: 5)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                X_synthetic, y_synthetic, X_combined, y_combined
        """

        # Input validation
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples")

        # Get class distribution
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        max_class_size = max(class_counts)

        # Determine target size per class
        if target_per_class is None:
            target_size = max_class_size
        else:
            if target_per_class < max_class_size:
                raise ValueError(
                    f"target_per_class ({target_per_class}) must be >= largest class size ({max_class_size})"
                )
            target_size = target_per_class

        # Display validation statistics
        print("=== Dataset Balancing Statistics ===")
        print("Original class distribution:")
        for cls, count in class_distribution.items():
            print(f"  Class {cls}: {count} samples")
        print(f"Target samples per class: {target_size}")
        print(f"Minimum class size threshold: {min_class_size}")

        # Filter classes and determine synthetic samples needed
        valid_classes = []
        skipped_classes = []
        synthetic_needed = {}

        for cls, count in class_distribution.items():
            if count < min_class_size:
                skipped_classes.append((cls, count))
                print(
                    f"Warning: Skipping class {cls} (only {count} samples, below threshold {min_class_size})"
                )
            else:
                valid_classes.append(cls)
                samples_to_generate = max(0, target_size - count)
                synthetic_needed[cls] = samples_to_generate

        if not valid_classes:
            raise ValueError("No classes meet the minimum size requirement")

        print("\nSynthetic samples to generate:")
        total_synthetic = 0
        for cls in valid_classes:
            count = synthetic_needed[cls]
            total_synthetic += count
            if count > 0:
                print(f"  Class {cls}: {count} synthetic samples")
            else:
                print(f"  Class {cls}: 0 synthetic samples (already at target)")

        if total_synthetic == 0:
            print("No synthetic samples needed - dataset is already balanced!")
            return (
                np.array([]).reshape(0, X_train.shape[1]),
                np.array([]),
                X_train.copy(),
                y_train.copy(),
            )

        # Scale the input data
        X_scaled = self.scaler.fit_transform(X_train)

        # Convert to tensors
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, device=self.device)

        # Generate synthetic samples for each class that needs them
        print(f"\nGenerating {total_synthetic} synthetic samples...")
        x_synthetic_list = []
        y_synthetic_list = []

        for cls in valid_classes:
            n_needed = synthetic_needed[cls]
            if n_needed > 0:
                print(f"\nGenerating {n_needed} samples for class {cls}...")
                x_synth_class = self._generate_samples_for_class(
                    cls, n_needed, x_train, y_train_tensor, X_scaled
                )
                y_synth_class = torch.full((n_needed,), cls, device=self.device)

                x_synthetic_list.append(x_synth_class)
                y_synthetic_list.append(y_synth_class)

        # Combine synthetic samples
        if x_synthetic_list:
            x_synthetic_combined = torch.cat(x_synthetic_list, dim=0)

            # Refine labels using TabPFN predictions
            print("Refining synthetic sample labels with TabPFN...")
            x_synthetic_np = x_synthetic_combined.detach().cpu().numpy()

            # Fit TabPFN classifier on valid classes only
            valid_mask = np.isin(y_train, valid_classes)
            X_train_valid = X_scaled[valid_mask]
            y_train_valid = y_train[valid_mask]

            clf = TabPFNClassifier(device=self.device)
            clf.fit(X_train_valid, y_train_valid)
            probs = clf.predict_proba(x_synthetic_np)
            y_synthetic_refined = torch.tensor(probs.argmax(axis=1), device=self.device)

            # Convert back to numpy and inverse transform
            X_synthetic = self.scaler.inverse_transform(x_synthetic_np)
            y_synthetic = y_synthetic_refined.cpu().numpy()

            # Map refined labels back to original class labels
            unique_valid_classes = np.unique(y_train_valid)
            y_synthetic_mapped = unique_valid_classes[y_synthetic]

        else:
            X_synthetic = np.array([]).reshape(0, X_train.shape[1])
            y_synthetic_mapped = np.array([])

        # Combine original and synthetic data
        X_combined = (
            np.vstack([X_train, X_synthetic])
            if len(X_synthetic) > 0
            else X_train.copy()
        )
        y_combined = (
            np.concatenate([y_train, y_synthetic_mapped])
            if len(y_synthetic_mapped) > 0
            else y_train.copy()
        )

        # Display final statistics
        print("\n=== Final Statistics ===")
        final_unique, final_counts = np.unique(y_combined, return_counts=True)
        final_distribution = dict(zip(final_unique, final_counts))

        print("Final combined class distribution:")
        for cls in sorted(final_distribution.keys()):
            count = final_distribution[cls]
            original_count = class_distribution.get(cls, 0)
            synthetic_count = count - original_count
            print(
                f"  Class {cls}: {count} total ({original_count} original + {synthetic_count} synthetic)"
            )

        if skipped_classes:
            print(f"\nSkipped classes: {[cls for cls, _ in skipped_classes]}")

        print("Dataset balancing completed!\n")
        print(
            "Note: The results represent an approximate balance that preserves data quality.\n"
        )

        return X_synthetic, y_synthetic_mapped, X_combined, y_combined

    def generate_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        balance_classes: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for classification.

        Args:
            X_train: np.ndarray
                Input features for training, shape (n_samples, n_features)
            y_train: np.ndarray
                Target labels for training, shape (n_samples,)
            n_samples: int
                Number of synthetic samples to generate
            balance_classes: bool
                Whether to balance classes in synthetic data (Default: True)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of synthetic features and labels
        """
        # Scale the input data
        X_scaled = self.scaler.fit_transform(X_train)

        # Convert to tensors
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(y_train, device=self.device)

        # Initialize synthetic data
        if balance_classes:
            classes = np.unique(y_train.cpu().numpy())
            n_per_class = n_samples // len(classes)
            x_synth_list = []
            y_synth_list = []

            for cls in classes:
                idx = np.where(y_train.cpu().numpy() == cls)[0]
                sample_idx = np.random.choice(idx, size=n_per_class)
                x_init = (
                    x_train[sample_idx]
                    + torch.randn(n_per_class, X_train.shape[1], device=self.device)
                    * 0.01
                )
                y_init = torch.full((n_per_class,), cls, device=self.device)

                x_synth_list.append(x_init)
                y_synth_list.append(y_init)

            x_synth = torch.cat(x_synth_list, dim=0)
            y_synth = torch.cat(y_synth_list, dim=0)
        else:
            x_synth = (
                torch.randn(n_samples, X_train.shape[1], device=self.device) * 0.01
            )
            y_synth = torch.randint(
                0, len(np.unique(y_train)), (n_samples,), device=self.device
            )

        # SGLD iterations
        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train, y_train)

            if step % 100 == 0:
                print(f"Step {step}/{self.n_sgld_steps}")

        # Generate final samples using TabPFN
        x_synth_np = x_synth.detach().cpu().numpy()
        x_train_np = x_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()

        # Fit TabPFN classifier
        clf = TabPFNClassifier(device=self.device)
        clf.fit(x_train_np, y_train_np)
        probs = clf.predict_proba(x_synth_np)

        # Refine labels based on TabPFN predictions
        y_synth = torch.tensor(probs.argmax(axis=1), device=self.device)

        # Convert back to numpy and inverse transform
        X_synth = self.scaler.inverse_transform(x_synth.detach().cpu().numpy())
        y_synth = y_synth.cpu().numpy()

        return X_synth, y_synth

    def generate_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        use_quantiles: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for regression.

        Args:
            X_train: np.ndarray
                Input features for training, shape (n_samples, n_features)
            y_train: np.ndarray
                Target values for training, shape (n_samples,)
            n_samples: int
                Number of synthetic samples to generate
            use_quantiles: bool
                Whether to use quantile regression for synthetic data (Default: True)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of synthetic features and target values
        """

        # Initialize regressor with appropriate preprocessing
        regressor = TabPFNRegressor(device=self.device)

        # Scale the input data
        X_scaled = self.scaler.fit_transform(X_train)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_scaled = (y_train - y_mean) / y_std

        # Convert to tensors for synthetic feature generation
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)

        # Initialize synthetic features using stratified sampling
        n_strata = 10
        y_strata = np.quantile(y_train, np.linspace(0, 1, n_strata + 1))
        x_synth_list = []
        samples_per_stratum = n_samples // n_strata

        for i in range(n_strata):
            # Get indices for this stratum
            mask = (y_train >= y_strata[i]) & (y_train <= y_strata[i + 1])
            stratum_indices = np.where(mask)[0]

            if len(stratum_indices) > 0:
                # Sample indices with replacement if needed
                sampled_indices = np.random.choice(
                    stratum_indices, size=samples_per_stratum
                )
                x_stratum = X_scaled[sampled_indices]

                # Add noise scaled by local variance
                stratum_std = np.std(x_stratum, axis=0)
                noise = np.random.normal(
                    0, stratum_std * 0.1, (samples_per_stratum, X_train.shape[1])
                )
                x_synth_list.append(x_stratum + noise)

        x_synth_init = np.vstack(x_synth_list)
        x_synth = torch.tensor(x_synth_init, device=self.device, dtype=torch.float32)

        # SGLD iterations with adaptive step size
        adaptive_step_size = self.sgld_step_size
        for step in range(self.n_sgld_steps):
            if step % 100 == 0:
                adaptive_step_size *= 0.9  # Gradually reduce step size

            x_synth = self._sgld_step(
                x_synth,
                torch.zeros(len(x_synth), device=self.device),
                x_train,
                torch.zeros_like(torch.tensor(y_scaled, device=self.device)),
            )

            if step % 100 == 0:
                print(f"Step {step}/{self.n_sgld_steps}")

        # Generate regression values using TabPFNRegressor
        x_synth_np = x_synth.detach().cpu().numpy()

        try:
            regressor.fit(X_scaled, y_scaled)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = regressor.predict(x_synth_np, output_type="full")

            if use_quantiles:
                quantiles = predictions["quantiles"]
                if not isinstance(quantiles, list):
                    quantiles = [quantiles]

                # Sample different quantiles for different ranges
                n_quantiles = len(quantiles)
                # Use more extreme quantiles for tails
                probs = np.abs(np.random.normal(0, 0.5, size=len(x_synth_np)))
                quantile_idx = (
                    (probs * n_quantiles).astype(int).clip(0, n_quantiles - 1)
                )
                y_synth = np.array(
                    [quantiles[i][j] for j, i in enumerate(quantile_idx)]
                )
            else:
                y_synth = np.array(predictions["median"])

            # Add small noise to prevent exact duplicates
            y_synth += np.random.normal(0, 0.01, size=len(y_synth))

        except Exception as e:
            print(f"Warning: Error in regression prediction: {str(e)}")
            print("Falling back to stratified sampling...")
            y_synth = np.random.normal(y_mean, y_std, size=len(x_synth_np))

        # Inverse transform the synthetic data
        X_synth = self.scaler.inverse_transform(x_synth_np)
        y_synth = y_synth * y_std + y_mean

        return X_synth, y_synth
