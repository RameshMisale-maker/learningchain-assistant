"""What happens after a user rates an answer (each step behind its own feature flag; never fails the request)."""
from __future__ import annotations

import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)
NEGATIVE_RATINGS = {"not_helpful", "stars_1", "stars_2"}


def on_feedback(message_id: int, rating: str, comment: str = "", username: str = "", session_id: str = "",
                product_id: str = "", stars: int | None = None, versions: dict | None = None) -> None:
    settings = get_settings()
    if stars is not None and stars <= 2:          # a low conversation rating always goes to the review queue
        try:
            from backend.services.review import on_low_rating

            on_low_rating(message_id, stars, comment, session_id, product_id, versions or {})
        except Exception:
            logger.exception("Could not queue the low rating of session %s for review", session_id)
    if settings.metrics_enabled:
        from backend.observability.metrics import record_feedback

        record_feedback(rating)
    if rating not in NEGATIVE_RATINGS:
        return
    if settings.response_cache_enabled:          # a disliked answer must not be served again from the cache
        try:
            from backend.services.response_cache import invalidate_message

            invalidate_message(message_id)
        except Exception:
            logger.exception("Could not invalidate the cached answer of message %s", message_id)
    if settings.review_queue_enabled and rating == "not_helpful":
        try:
            from backend.services.review import on_thumbs_down

            on_thumbs_down(message_id, comment)
        except Exception:
            logger.exception("Could not queue message %s for review", message_id)
