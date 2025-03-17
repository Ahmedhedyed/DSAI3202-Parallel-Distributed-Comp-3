from collections.abc import Iterator
from typing import Protocol

import gdb

def get_basic_type(type_: gdb.Type) -> gdb.Type: ...
def has_field(type_: gdb.Type, field: str) -> bool: ...
def make_enum_dict(enum_type: gdb.Type) -> dict[str, int]: ...
def deep_items(type_: gdb.Type) -> Iterator[tuple[str, gdb.Field]]: ...
def get_type_recognizers() -> list[_TypeRecognizer]: ...
def apply_type_recognizers(recognizers: list[_TypeRecognizer], type_obj: gdb.Type) -> str | None: ...
def register_type_printer(locus: gdb.Objfile | gdb.Progspace | None, printer: _TypePrinter) -> None: ...

class _TypePrinter(Protocol):
    enabled: bool
    name: str

    def instantiate(self) -> _TypeRecognizer | None: ...

class _TypeRecognizer(Protocol):
    def recognize(self, type: gdb.Type, /) -> str | None: ...

class TypePrinter:
    enabled: bool
    name: str

    def __init__(self, name: str) -> None: ...
    def instantiate(self) -> _TypeRecognizer | None: ...
