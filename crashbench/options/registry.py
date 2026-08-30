"""Separate deployable and diagnostic option catalogs with loader firewalls."""

from __future__ import annotations

from typing import Iterable

from .base import CatalogKind, OptionSpec


class OptionCatalog:
    def __init__(self, kind: CatalogKind, specs: Iterable[OptionSpec]):
        self.kind = CatalogKind(kind)
        self._specs: dict[str, OptionSpec] = {}
        for spec in specs:
            if spec.catalog_kind != self.kind:
                raise ValueError(
                    f"{spec.canonical_id} cannot enter {self.kind.value} catalog"
                )
            if spec.option_id in self._specs:
                raise ValueError(f"duplicate option_id in catalog: {spec.option_id}")
            self._specs[spec.option_id] = spec
        if not self._specs:
            raise ValueError("option catalog cannot be empty")

    def get(self, option_id: str, *, consumer_kind: CatalogKind) -> OptionSpec:
        consumer = CatalogKind(consumer_kind)
        if consumer == CatalogKind.DEPLOYABLE and self.kind != CatalogKind.DEPLOYABLE:
            raise PermissionError("deployable consumer cannot load diagnostic oracle catalog")
        try:
            return self._specs[option_id]
        except KeyError as exc:
            raise KeyError(f"unknown {self.kind.value} option_id: {option_id}") from exc

    def canonical_ids(self) -> tuple[str, ...]:
        return tuple(self._specs[key].canonical_id for key in sorted(self._specs))

    def assert_disjoint(self, other: "OptionCatalog") -> None:
        overlap = set(self._specs) & set(other._specs)
        if overlap:
            raise ValueError(f"deployable/diagnostic option IDs overlap: {sorted(overlap)}")
