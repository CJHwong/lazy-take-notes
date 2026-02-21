"""Port: template loader."""

from __future__ import annotations

from typing import Protocol

from lazy_take_notes.l1_entities.template import SessionTemplate, TemplateMetadata


class TemplateLoader(Protocol):  # pragma: no cover -- abstract Protocol; never instantiated directly
    """Abstract template loader."""

    def load(self, template_ref: str) -> SessionTemplate:
        """Load a template by name or file path."""
        ...

    def list_templates(self) -> list[TemplateMetadata]:
        """List available templates (built-in and user)."""
        ...
