"""Per-request product: resolution (explicit id / legacy assistant alias / domain / API key / default) and a context
variable so any module can read `current_product()` without passing it through every call."""
from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from backend.products.config import ProductConfig, ProductConfigError, default_product_id, get_product, load_products

_current: contextvars.ContextVar[ProductConfig | None] = contextvars.ContextVar("current_product", default=None)


def current_product() -> ProductConfig:
    """The product of the request being handled, or the default product outside a request."""
    return _current.get() or get_product()


@contextmanager
def use_product(product: ProductConfig | str | None) -> Iterator[ProductConfig]:
    # None keeps the product already in effect (or the default outside any request)
    cfg = product if isinstance(product, ProductConfig) else (get_product(product) if product else current_product())
    token = _current.set(cfg)
    try:
        yield cfg
    finally:
        _current.reset(token)


product_context = use_product


def resolve_product(product: str | None = None, assistant: str | None = None, host: str | None = None,
                    api_key_product: str | None = None) -> ProductConfig:
    """Order: explicit product id -> legacy `assistant` alias -> request domain -> API key -> default.

    With PRODUCT_CONFIG_ENABLED off, every request uses the default product (today's single combined assistant);
    the legacy `assistant` value still narrows the knowledge bases through the product's `scopes`."""
    from config.settings import get_settings

    products = load_products()
    if not get_settings().product_config_enabled:
        return get_product()
    if product:
        if product not in products:
            raise ProductConfigError(f"unknown product '{product}'")
        return products[product]
    if assistant:
        for cfg in products.values():
            if assistant in cfg.aliases:
                return cfg
    if host:
        bare = host.split(":")[0].lower()
        for cfg in products.values():
            if bare in (d.lower() for d in cfg.domains):
                return cfg
    if api_key_product and api_key_product in products:
        return products[api_key_product]
    return products[default_product_id()]
