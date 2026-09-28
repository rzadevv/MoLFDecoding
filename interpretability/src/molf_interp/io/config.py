"""YAML config loading and Pydantic schema definitions."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base class for all config schemas.

    All subclasses inherit strict validation and immutability.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class HelloConfig(BaseConfig):
    """Config for the hello CLI command.

    Attributes:
        name: Name to greet.
        greeting: Greeting word to prepend.
    """

    name: str
    greeting: str = "hello"


def load_config(path: Path, schema: type[T]) -> T:
    """Load a YAML file and validate it against a Pydantic schema.

    Args:
        path: Path to the YAML config file.
        schema: Pydantic model class to validate against.

    Returns:
        Validated config instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the YAML does not match the schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return schema.model_validate(raw)
