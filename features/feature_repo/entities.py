"""
Entity Definitions for Feast Feature Store

Entities are the primary keys used to look up features.
"""

from feast import Entity
from feast.types import String


# User entity - represents a customer/user
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="A user or customer in the system",
)


# Product entity - represents a product
product = Entity(
    name="product",
    join_keys=["product_id"],
    description="A product in the catalog",
)
