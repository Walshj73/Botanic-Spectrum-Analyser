"""Small explicit registry used by methods and dataset adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

FactoryT = TypeVar("FactoryT", bound=Callable[..., object])


class Registry(Generic[FactoryT]):
    def __init__(self, category: str) -> None:
        self._category = category
        self._factories: dict[str, FactoryT] = {}

    def register(self, identifier: str, factory: FactoryT) -> None:
        if not identifier:
            raise ValueError("registry identifier must not be empty")
        if identifier in self._factories:
            raise ValueError(f"duplicate {self._category} identifier: {identifier}")
        self._factories[identifier] = factory

    def get(self, identifier: str) -> FactoryT:
        try:
            return self._factories[identifier]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(
                f"unknown {self._category} {identifier!r}; available: {available}"
            ) from exc

    def identifiers(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
