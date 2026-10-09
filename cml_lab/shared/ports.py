"""Coordination contracts shared by dataset ownership and execution intent."""

from typing import ContextManager, Protocol


class MutationGuard(Protocol):
    def hold(self) -> ContextManager[None]: ...
