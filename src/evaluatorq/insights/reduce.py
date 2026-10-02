"""3D UMAP reduction of embedding vectors, for the dashboard's cluster map.

Pure numpy/umap — no LLM or network calls, so there is no retry layer here. `umap` is
imported inside `reduce_3d` (not at module scope) since it is a slow-importing dependency
behind the `insights` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, n - 1),
            metric='cosine',
            random_state=random_state,
        )
        coordinates = np.asarray(reducer.fit_transform(vectors))
        if coordinates.shape != (n, 3) or not np.isfinite(coordinates).all():
            raise ValueError(f'UMAP returned invalid coordinates with shape {coordinates.shape}')
        return coordinates
    except Exception as exc:  # noqa: BLE001 - reduction is optional and must degrade to no map
        logger.warning('reduce_3d: UMAP failed; omitting map coordinates: {}', exc)
        return None
