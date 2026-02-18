"""Template Pydantic models — pure data, no I/O."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TemplateMetadata(BaseModel):
    name: str = ''
    description: str = ''
    locale: str = ''


class QuickAction(BaseModel):
    key: str
    label: str
    description: str = ''
    prompt_template: str


class SessionTemplate(BaseModel):
    metadata: TemplateMetadata = Field(default_factory=TemplateMetadata)
    system_prompt: str = ''
    digest_user_template: str = ''
    final_user_template: str = ''
    whisper_prompt: str = ''
    quick_actions: list[QuickAction] = Field(default_factory=list)
