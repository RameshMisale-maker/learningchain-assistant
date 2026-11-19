"""Product configuration: everything product-specific (names, glossary, replies, sources, models, UI text) lives in
`config/products/<product_id>.yaml`; the pipeline code reads it through `current_product()`."""
from backend.products.config import (ProductConfig, ProductConfigError, default_product_id, get_product,
                                     load_products, product_for_kb, reset_products)
from backend.products.context import current_product, product_context, resolve_product, use_product

__all__ = ["ProductConfig", "ProductConfigError", "current_product", "default_product_id", "get_product",
           "load_products", "product_context", "product_for_kb", "reset_products", "resolve_product", "use_product"]
