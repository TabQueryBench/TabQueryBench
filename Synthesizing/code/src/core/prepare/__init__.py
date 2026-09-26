"""Dataset preparation: profile.yaml + raw CSV -> runner-ready splits and field registry."""

from .pipeline import PrepareError, prepare_dataset, prepare_many
from .profile import SCHEMA_VERSION, DatasetProfile, ProfileError, draft_profile, load_profile

__all__ = [
    "SCHEMA_VERSION",
    "DatasetProfile",
    "PrepareError",
    "ProfileError",
    "draft_profile",
    "load_profile",
    "prepare_dataset",
    "prepare_many",
]
