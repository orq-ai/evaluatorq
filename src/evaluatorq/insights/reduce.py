"""UMAP layout generation during Insights analysis, for saved 3D and 2D maps.

Pure numpy/umap — no LLM or network calls, so there is no retry layer here. `umap` is
imported inside the analysis functions (not at module scope) since it is a slow-importing dependency
behind the `insights` extra.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Numba's workqueue backend cannot safely fit UMAP from concurrent analysis threads.
_UMAP_LOCK = Lock()


def reduce_3d(vectors: NDArray[Any], *, random_state: int = 42) -> NDArray[Any] | None:
    """Reduce `vectors` to 3D coordinates via UMAP (cosine metric).

    Returns `None` with a warning when the input cannot produce stable coordinates
    or UMAP fails; coordinate generation is optional for a dimension.
    """
    import numpy as np

    if vectors.ndim != 2:
        logger.warning('reduce_3d: skipping UMAP for malformed vector matrix with shape {}', vectors.shape)
        return None

    n = vectors.shape[0]
    if n < 5:
        logger.warning('reduce_3d: skipping UMAP with only {} point(s) (need >= 5)', n)
        return None
    try:
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            logger.warning('reduce_3d: skipping UMAP because vectors contain non-finite or zero rows')
            return None
        if np.allclose(vectors, vectors[0]):
            logger.warning('reduce_3d: skipping UMAP because all vectors are identical')
            return None
        import umap

        with _UMAP_LOCK:
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(15, n - 1),
                metric='cosine',
                random_state=random_state,
                n_jobs=1,
            )
            coordinates = np.asarray(reducer.fit_transform(vectors))
        if coordinates.shape != (n, 3) or not np.isfinite(coordinates).all():
            raise ValueError(f'UMAP returned invalid coordinates with shape {coordinates.shape}')
        return coordinates
    except Exception as exc:  # noqa: BLE001 - reduction is optional and must degrade to no map
        logger.warning('reduce_3d: UMAP failed; omitting map coordinates: {}', exc)
        return None


def project_2d(coordinates: NDArray[Any], *, random_state: int = 7) -> NDArray[Any] | None:
    """Generate the review layout from saved 3D coordinates before the run is persisted.

    Return None with a warning if fewer than five finite 3D points are available or projection fails.
    """
    import numpy as np

    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) < 5:
        logger.warning('project_2d: skipping UMAP for unusable coordinate matrix with shape {}', coordinates.shape)
        return None
    if not np.isfinite(coordinates).all():
        logger.warning('project_2d: skipping UMAP because coordinates contain non-finite values')
        return None
    try:
        import umap

        with _UMAP_LOCK:
            projected = np.asarray(
                umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, len(coordinates) - 1),
                    metric='euclidean',
                    random_state=random_state,
                    n_jobs=1,
                ).fit_transform(coordinates)
            )
        if projected.shape != (len(coordinates), 2) or not np.isfinite(projected).all():
            raise ValueError(f'UMAP returned invalid coordinates with shape {projected.shape}')
        return projected
    except Exception as exc:  # noqa: BLE001 - a failed map must not discard the analysis
        logger.warning('project_2d: UMAP failed; omitting review map coordinates: {}', exc)
        return None
