"""Input validation helper shared by all model implementations.

Each model calls ``validate_input`` at the top of ``predict()`` with its
own manifest spec.  This keeps validation logic in one place rather than
duplicated across model families.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from exg_emotion.core.exceptions import InputValidationError


def validate_input(
    samples: np.ndarray,
    expected_channels: int,
    expected_samples: int,
    channel_names: Optional[Sequence[str]] = None,
    expected_channel_names: Optional[Sequence[str]] = None,
) -> None:
    """Validate a signal window against a model's declared input spec.

    Parameters
    ----------
    samples:
        The array passed to ``predict()``.
    expected_channels:
        Number of channels declared in the manifest.
    expected_samples:
        Number of samples per window declared in the manifest.
    channel_names:
        Optional channel names supplied by the caller (may be ``None``).
    expected_channel_names:
        Optional ordered channel names from the manifest.  Compared to
        *channel_names* only when both are provided.

    Raises
    ------
    InputValidationError
        If the array has the wrong shape, dtype-convertible issues, or
        non-finite values.
    """
    if not isinstance(samples, np.ndarray):
        raise InputValidationError(
            f"samples must be a numpy ndarray, got {type(samples).__name__}",
            expected="numpy.ndarray",
            got=type(samples).__name__,
        )

    if samples.ndim != 2:
        raise InputValidationError(
            f"samples must be 2-D [n_channels, n_samples], got ndim={samples.ndim}",
            expected="ndim=2",
            got=f"ndim={samples.ndim}",
        )

    n_ch, n_samp = samples.shape

    if n_ch != expected_channels:
        raise InputValidationError(
            f"Expected {expected_channels} channels, got {n_ch}",
            expected=expected_channels,
            got=n_ch,
        )

    if n_samp != expected_samples:
        raise InputValidationError(
            f"Expected {expected_samples} samples per window, got {n_samp}",
            expected=expected_samples,
            got=n_samp,
        )

    if not np.all(np.isfinite(samples)):
        n_bad = int(np.sum(~np.isfinite(samples)))
        raise InputValidationError(
            f"samples contains {n_bad} non-finite value(s) (NaN or Inf). "
            "Interpolate or drop the window before calling predict().",
        )

    if channel_names is not None and expected_channel_names is not None:
        if list(channel_names) != list(expected_channel_names):
            raise InputValidationError(
                f"Channel name mismatch.\n"
                f"  Expected: {expected_channel_names}\n"
                f"  Got:      {list(channel_names)}",
                expected=list(expected_channel_names),
                got=list(channel_names),
            )
