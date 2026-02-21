"""Template Pydantic models — pure data, no I/O."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TemplateMetadata(BaseModel):
    name: str = ''
    description: str = ''
    locale: str = ''
    key: str = ''  # file key (set by loader, not stored in YAML)


class QuickAction(BaseModel):
    label: str
    description: str = ''
    prompt_template: str


class SessionTemplate(BaseModel):
    metadata: TemplateMetadata = Field(default_factory=TemplateMetadata)
    system_prompt: str = ''
    digest_user_template: str = ''
    final_user_template: str = ''
    recognition_hints: list[str] = Field(default_factory=list)
    quick_actions: list[QuickAction] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_quick_actions_count(self) -> SessionTemplate:
        if len(self.quick_actions) > 5:
            raise ValueError(f'At most 5 quick_actions allowed, got {len(self.quick_actions)}')
        return self
