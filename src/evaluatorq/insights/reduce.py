"""3D UMAP reduction of embedding vectors, for the dashboard's cluster map.

Pure numpy/umap — no LLM or network calls, so there is no retry layer here. `umap` is
imported inside `reduce_3d` (not at module scope) since it is a slow-importing dependency
behind the `insights` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


def reduce_3d(vectors: NDArray[Any], *, random_state: int = 42) -> NDArray[Any] | None:
    """Reduce `vectors` to 3D coordinates via UMAP (cosine metric).

    Returns `None` with a `logger.warning` when there are fewer than 5 points — too few
    for a stable UMAP embedding.
    """
    n = vectors.shape[0]
    if n < 5:
        logger.warning('reduce_3d: skipping UMAP with only {} point(s) (need >= 5)', n)
        return None

    import umap

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(15, n - 1),
        metric='cosine',
        random_state=random_state,
    )
    return cast('NDArray[Any]', reducer.fit_transform(vectors))
