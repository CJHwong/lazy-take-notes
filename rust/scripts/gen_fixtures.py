"""Record the Python outputs that the Rust port must reproduce byte for byte.

Run from the repo root: uv run python rust/scripts/gen_fixtures.py
"""

from __future__ import annotations

import asyncio
import json
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.template import QuickAction, SessionTemplate, TemplateMetadata
from lazy_take_notes.l2_use_cases import transcribe_audio_use_case as tau
from lazy_take_notes.l2_use_cases.generate_label_use_case import sanitize_label
from lazy_take_notes.l2_use_cases.ports.llm_client import ChatResponse
from lazy_take_notes.l2_use_cases.template_builder_use_case import TemplateBuildUseCase
from lazy_take_notes.l2_use_cases.utils.prompt_builder import (
    build_compact_user_message,
    build_digest_prompt,
    build_label_prompt,
    build_quick_action_prompt,
)
from lazy_take_notes.l2_use_cases.utils.template_diff import diff_templates
from lazy_take_notes.l2_use_cases.utils.template_validator import validate_template

OUT = Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
SCHEMA_OUT = Path(__file__).resolve().parent.parent / 'src' / 'usecases' / 'session_template_schema.json'


def lcg_audio(seed: int, seconds: float, amplitude: float, syllables: bool) -> np.ndarray:
    """Uniform noise from a 32-bit LCG, optionally gated at syllable rate. Rust rebuilds the same samples."""
    count = int(seconds * 16000)
    state = seed
    out = np.zeros(count, dtype=np.float32)
    for index in range(count):
        state = (1664525 * state + 1013904223) % 2**32
        value = state / 2**32 * 2.0 - 1.0
        if syllables and (index / 16000) % 0.5 >= 0.4:
            value = 0.0
        out[index] = np.float32(value * amplitude)
    return out


def templates() -> dict:
    """The built-in templates as the loader returns them, with the key set to the file name."""
    loaded = {}
    for path in sorted(resources.files('lazy_take_notes').joinpath('templates').iterdir(), key=lambda p: p.name):
        if not path.name.endswith('.yaml'):
            continue
        template = SessionTemplate.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')) or {})
        template.metadata.key = path.name.removesuffix('.yaml')
        loaded[template.metadata.key] = template.model_dump()
    return loaded


def prompts(loaded: dict) -> list[dict]:
    cases = []
    for name, data in loaded.items():
        tmpl = SessionTemplate.model_validate(data)
        for is_final in (False, True):
            for context in ('', 'John = speaker A\n  ', '   \n  '):
                for full in ('', 'Full session text'):
                    cases.append(
                        {
                            'template': name,
                            'buffer': ['Line 1', '第二行 中文'],
                            'is_final': is_final,
                            'user_context': context,
                            'full_transcript': full,
                            'out': build_digest_prompt(
                                tmpl,
                                ['Line 1', '第二行 中文'],
                                is_final=is_final,
                                full_transcript=full,
                                user_context=context,
                            ),
                        }
                    )
    return cases


def quick_actions(loaded: dict) -> list[dict]:
    cases = []
    for data in loaded.values():
        for action in data['quick_actions']:
            for digest, recent, context in (('', '', ''), ('## Notes', 'a\nb', 'Huss = Hoss')):
                cases.append(
                    {
                        'prompt_template': action['prompt_template'],
                        'digest': digest,
                        'recent': recent,
                        'user_context': context,
                        'out': build_quick_action_prompt(
                            action['prompt_template'], digest, recent, user_context=context
                        ),
                    }
                )
    return cases


def validation() -> list[dict]:
    base = {
        'metadata': {'name': 'Test', 'locale': 'en'},
        'system_prompt': 'You are a meeting assistant.',
        'digest_user_template': 'Lines ({line_count}):\n{new_lines}\n{user_context}',
        'final_user_template': 'Final:\n{full_transcript}',
        'quick_actions': [{'label': 'Summary', 'prompt_template': '{digest_markdown}'}],
    }
    variants = [
        {},
        {'digest_user_template': 'Hello {bogus_var} {zeta}'},
        {'digest_user_template': '{full_transcript}'},
        {'digest_user_template': '{line_count} {unclosed'},
        {'digest_user_template': '{line_count} {'},
        {'digest_user_template': 'a } b'},
        {'digest_user_template': '{{literal}} {line_count}'},
        {'digest_user_template': '{}'},
        {'final_user_template': 'Hello {nonexistent}'},
        {
            'quick_actions': [
                {'label': 'OK', 'prompt_template': '{digest_markdown}'},
                {'label': 'B', 'prompt_template': '{nope}'},
            ]
        },
        {'system_prompt': '   \n  '},
    ]
    cases = []
    for variant in variants:
        data = {**base, **variant}
        result = validate_template(SessionTemplate.model_validate(data))
        cases.append({'template': data, 'errors': result.errors, 'text': str(result)})
    return cases


def diffs() -> list[dict]:
    def make(**changes: object) -> SessionTemplate:
        base = SessionTemplate(
            metadata=TemplateMetadata(name='N', description='D', locale='en'),
            system_prompt='S',
            recognition_hints=['GoFreight', "O'Brien", '貨代'],
            quick_actions=[
                QuickAction(label='A', prompt_template='a'),
                QuickAction(label='B', prompt_template='b'),
            ],
        )
        return base.model_copy(update=changes)

    old = make()
    news = [
        make(),
        make(system_prompt='S2', metadata=TemplateMetadata(name='N2', description='D', locale='zh-TW', key='x')),
        make(recognition_hints=['GoFreight', 'say "hi"', '新詞']),
        make(quick_actions=[QuickAction(label='B', prompt_template='b'), QuickAction(label='A', prompt_template='a')]),
        make(
            quick_actions=[QuickAction(label='A', prompt_template='a2'), QuickAction(label="C'", prompt_template='c')]
        ),
    ]
    return [{'old': old.model_dump(), 'new': new.model_dump(), 'out': diff_templates(old, new)} for new in news]


def labels() -> list[dict]:
    raws = [
        '"Sprint Planning Review"',
        "  'weekly sync'  ",
        'Team Standup: Q3 Roadmap & Budget!!',
        'one two three four five six seven',
        '中文 標籤',
        '__already_snake__',
        '',
    ]
    return [{'raw': raw, 'out': sanitize_label(raw)} for raw in raws]


def cleaning() -> dict:
    texts = [
        '*clap*',
        '*sad music*',
        '[BLANK_AUDIO]',
        '(laughs)',
        '♪',
        '.',
        '...',
        ' - ',
        '，。',
        'Real words.',
        '好的。',
        'We agreed. ' + 'And you can write the search plan by next Friday. ' * 51,
        'we agreed ' + 'and you can write the search plan by next friday ' * 20,
        'No. No. I mean Friday.',
        '我們明天再討論這個問題吧。我們明天再討論這個問題吧。我們明天再討論這個問題吧。',
        '"We\'re going to show you a couple of things." "We\'re going to show you a couple of things."',
    ]
    collapse = [{'text': t, 'out': tau._collapse_loops(t.strip())} for t in texts]
    not_speech = [{'text': t, 'out': tau._NOT_SPEECH.fullmatch(t.strip()) is not None} for t in texts]
    return {'collapse': collapse, 'not_speech': not_speech}


class Clip(NamedTuple):
    seed: int
    seconds: float
    amplitude: float
    syllables: bool
    lead_s: float


def gate() -> list[dict]:
    clips = [
        Clip(seed=1, seconds=10.0, amplitude=0.02, syllables=False, lead_s=0.0),
        Clip(seed=2, seconds=5.0, amplitude=0.1, syllables=True, lead_s=0.0),
        Clip(seed=3, seconds=5.0, amplitude=0.02, syllables=True, lead_s=20.0),
        Clip(seed=4, seconds=4.0, amplitude=0.05, syllables=True, lead_s=6.0),
    ]
    cases = []
    for clip in clips:
        speech = lcg_audio(clip.seed, clip.seconds, clip.amplitude, clip.syllables)
        audio = np.concatenate([np.zeros(int(clip.lead_s * 16000), dtype=np.float32), speech])
        flags = tau.voiced_frames(audio, 0.01)
        spans = [(0.0, 2.0), (1.0, 3.0), (0.0, 24.0), (4.0, 9.0), (6.05, 7.0)]
        cases.append(
            {
                **clip._asdict(),
                'voiced': [bool(f) for f in flags],
                'voiced_seconds': tau.voiced_seconds(audio, 0.01),
                'speech_start': [[s, e, tau.speech_start(flags, s, e)] for s, e in spans],
            }
        )
    return cases


class _RecordingLLM:
    """Keeps the messages of the last chat call; replies with a question so no template is parsed."""

    def __init__(self) -> None:
        self.sent: list[ChatMessage] = []

    async def chat(self, model: str, messages: list[ChatMessage]) -> ChatResponse:
        self.sent = messages
        return ChatResponse(content='?')

    async def chat_single(self, model: str, prompt: str) -> str:
        return '?'

    def check_connectivity(self) -> tuple[bool, str]:
        return True, ''

    def check_models(self, models: list[str]) -> list[str]:
        return []


def system_prompt_sent(use_case: TemplateBuildUseCase) -> str:
    """The system prompt the builder sends on its first turn."""
    asyncio.run(use_case.generate('hi', 'm', []))
    llm = use_case.llm_client
    if not isinstance(llm, _RecordingLLM):
        raise TypeError('system_prompt_sent needs a use case built on _RecordingLLM')
    return llm.sent[0].content


def builder(loaded: dict) -> dict:
    example = SessionTemplate.model_validate(loaded['default_en'])
    current = SessionTemplate.model_validate(
        {**loaded['default_zh_tw'], 'metadata': {**loaded['default_zh_tw']['metadata'], 'key': 'default_zh_tw'}}
    )
    return {
        'example': example.model_dump(),
        'current': current.model_dump(),
        'create_prompt': system_prompt_sent(TemplateBuildUseCase(llm_client=_RecordingLLM(), example_template=example)),
        'edit_prompt': system_prompt_sent(
            TemplateBuildUseCase(llm_client=_RecordingLLM(), example_template=example, base_template=current)
        ),
    }


def misc() -> dict:
    return {
        'label_prompt': [
            {
                'name': 'Daily Standup',
                'description': 'Quick sync meeting',
                'digest': '## Notes\nAll good.',
                'out': build_label_prompt('Daily Standup', 'Quick sync meeting', '## Notes\nAll good.'),
            },
            {'name': 'Default', 'description': '', 'digest': 'd', 'out': build_label_prompt('Default', '', 'd')},
        ],
        'compact': {'markdown': '## Topic\n中文', 'out': build_compact_user_message('## Topic\n中文')},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = templates()
    fixtures = {
        'templates': loaded,
        'prompts': prompts(loaded),
        'quick_actions': quick_actions(loaded),
        'validation': validation(),
        'diffs': diffs(),
        'labels': labels(),
        'cleaning': cleaning(),
        'gate': gate(),
        'builder': builder(loaded),
        'misc': misc(),
    }
    for name, value in fixtures.items():
        (OUT / f'{name}.json').write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    SCHEMA_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_OUT.write_text(json.dumps(SessionTemplate.model_json_schema(), indent=2) + '\n', encoding='utf-8')
    print(f'wrote {len(fixtures)} fixture files to {OUT}')


if __name__ == '__main__':
    main()
