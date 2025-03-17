from typing import Any, Callable
from typing_extensions import LiteralString

class Undecidable(ValueError): ...

def filldedent(s, w=..., **kwargs) -> str: ...
def strlines(s, c=..., short=...) -> str | LiteralString: ...
def rawlines(s) -> str | LiteralString: ...

ARCH = ...
HASH_RANDOMIZATION = ...
_debug_tmp: list[str] = ...
_debug_iter = ...

def debug_decorator(func) -> Callable[..., Any]: ...
def debug(*args) -> None: ...
def debugf(string, args) -> None: ...
def find_executable(executable, path=...) -> str | None: ...
def func_name(x, short=...) -> str | Any: ...
def replace(string, *reps) -> str: ...
def translate(s, a, b=..., c=...) -> str: ...
def ordinal(num) -> str: ...
def as_int(n, strict=...) -> int: ...
