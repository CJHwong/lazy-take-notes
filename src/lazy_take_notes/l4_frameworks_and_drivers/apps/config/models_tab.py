"""Models page: text generation, speech models, language overrides, and speakers."""

from __future__ import annotations

import subprocess  # noqa: S404 -- used to open the cache folder, not a shell command
import sys
import threading
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, ProgressBar, Rule, Static

from lazy_take_notes.l1_entities.model_asset import AssetProgress, AssetState, AssetStatus
from lazy_take_notes.l2_use_cases.ports.asset_store import ModelAdminStore
from lazy_take_notes.l3_interface_adapters.gateways import model_catalog
from lazy_take_notes.l3_interface_adapters.gateways.download_engine import DownloadCancelledError
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.dialogs import RemoveModelModal
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.fields import (
    FieldRow,
    SelectRow,
    discover_providers,
)
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.sections import (
    LanguageModelsEditor,
    SettingsSection,
    model_uri,
    section_title,
)
from lazy_take_notes.l4_frameworks_and_drivers.container import (
    BUILTIN_LLM_PROVIDERS,
    BUILTIN_TRANSCRIPTION_PROVIDERS,
    LLM_PROVIDERS_GROUP,
    TRANSCRIPTION_PROVIDERS_GROUP,
)

_STATUS_TEXT = {
    AssetStatus.READY: 'Ready',
    AssetStatus.MISSING: 'Missing',
    AssetStatus.DOWNLOADING: 'Downloading',
    AssetStatus.CORRUPT: 'Damaged',
    AssetStatus.UNSUPPORTED: 'Unsupported',
}


def _megabytes(size: int) -> str:
    if size >= 1024**3:
        return f'{size / 1024**3:.1f} GB'
    return f'{size / (1024 * 1024):.0f} MB'


class ModelChoiceRow(Vertical):
    """One transcript model. Click the row to select it, or use its action."""

    DEFAULT_CSS = """
    ModelChoiceRow {
        height: auto;
        padding: 0;
        margin: 0 0 1 0;
    }
    ModelChoiceRow .choice-line {
        height: 1;
    }
    ModelChoiceRow .choice-mark {
        width: 2;
        color: $accent;
    }
    ModelChoiceRow .choice-label {
        width: 30;
        text-style: bold;
    }
    ModelChoiceRow .choice-size {
        width: 9;
        color: $text-muted;
    }
    ModelChoiceRow .choice-status {
        width: 12;
    }
    ModelChoiceRow .choice-status.ready {
        color: $success;
    }
    ModelChoiceRow .choice-status.missing {
        color: $warning;
    }
    ModelChoiceRow .choice-status.corrupt {
        color: $error;
    }
    ModelChoiceRow .choice-status.downloading {
        color: $accent;
    }
    ModelChoiceRow .choice-progress {
        margin: 0 0 0 2;
    }
    ModelChoiceRow Button {
        margin-left: 1;
    }
    """

    def __init__(self, asset, state: AssetState, selected: bool) -> None:
        super().__init__(id=f'choice-{asset.id}')
        self.asset = asset
        self._state = state
        self._selected = selected

    def compose(self) -> ComposeResult:
        with Horizontal(classes='choice-line'):
            yield Static('●' if self._selected else '○', classes='choice-mark')
            yield Static(self.asset.label, classes='choice-label')
            yield Static(_megabytes(self.asset.size), classes='choice-size')
            yield Static('', id=f'status-{self.asset.id}', classes='choice-status')
            yield Button('Download', id=f'download-{self.asset.id}')
            yield Button('Remove', id=f'remove-{self.asset.id}')
            yield Button('Cancel', id=f'cancel-{self.asset.id}', variant='error')
        yield ProgressBar(total=100, show_eta=False, id=f'progress-{self.asset.id}', classes='choice-progress')

    def on_mount(self) -> None:
        self.set_state(self._state)

    def on_click(self, event) -> None:
        if isinstance(event.widget, Button):
            return
        self.post_message(ModelChooser.Selected(self.asset.id))

    def set_state(self, state: AssetState) -> None:
        self._state = state
        status = state.status
        label = self.query_one(f'#status-{self.asset.id}', Static)
        label.update(_STATUS_TEXT[status])
        for name in ('ready', 'missing', 'corrupt', 'downloading', 'unsupported'):
            label.remove_class(name)
        label.add_class(status.value)
        self._action('download').display = status in (AssetStatus.MISSING, AssetStatus.CORRUPT)
        self._action('remove').display = status in (AssetStatus.READY, AssetStatus.CORRUPT)
        self._action('cancel').display = status is AssetStatus.DOWNLOADING
        self.query_one(f'#progress-{self.asset.id}', ProgressBar).display = status is AssetStatus.DOWNLOADING

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.query_one('.choice-mark', Static).update('●' if selected else '○')

    def mark_downloading(self) -> None:
        self.set_state(AssetState(self.asset, AssetStatus.DOWNLOADING))

    def set_progress(self, progress: AssetProgress) -> None:
        bar = self.query_one(f'#progress-{self.asset.id}', ProgressBar)
        total = progress.total or self.asset.size or 1
        bar.display = True
        bar.update(total=total, progress=progress.downloaded)
        label = self.query_one(f'#status-{self.asset.id}', Static)
        label.update(f'{_megabytes(progress.downloaded)} / {_megabytes(total)}')

    def _action(self, name: str) -> Button:
        return self.query_one(f'#{name}-{self.asset.id}', Button)


class ModelChooser(Vertical):
    """The transcript model list. The selected row is the default model."""

    DEFAULT_CSS = """
    ModelChooser {
        height: auto;
    }
    """

    class Selected(Message):
        def __init__(self, asset_id: str) -> None:
            super().__init__()
            self.asset_id = asset_id

    def __init__(self, assets, selected_uri: str) -> None:
        super().__init__()
        self._assets = assets
        self._selected_id = self._match(selected_uri)

    def compose(self) -> ComposeResult:
        for asset in self._assets:
            yield ModelChoiceRow(asset, AssetState(asset, AssetStatus.MISSING), asset.id == self._selected_id)

    def _match(self, uri: str) -> str | None:
        for asset in self._assets:
            if model_uri(asset) == uri:
                return asset.id
        return None

    @on(Selected)
    def _on_selected(self, message: Selected) -> None:
        self.select(message.asset_id)

    def select(self, asset_id: str) -> None:
        self._selected_id = asset_id
        for row in self.query(ModelChoiceRow):
            row.set_selected(row.asset.id == asset_id)

    def select_by_uri(self, uri: str) -> None:
        asset_id = self._match(uri)
        if asset_id is not None:
            self.select(asset_id)

    @property
    def selected_uri(self) -> str:
        for asset in self._assets:
            if asset.id == self._selected_id:
                return model_uri(asset)
        return ''

    def set_state(self, state: AssetState) -> None:
        for row in self.query(ModelChoiceRow):
            if row.asset.id == state.asset.id:
                row.set_state(state)
                return


class ModelsTab(SettingsSection):
    """Text generation, speech models, language overrides, and speaker labels."""

    DEFAULT_CSS = """
    ModelsTab .button-row {
        height: auto;
        margin: 0 0 1 0;
    }
    ModelsTab .button-row Button {
        margin-right: 1;
    }
    ModelsTab .subsection-title {
        height: 1;
        text-style: bold;
        margin: 0 0 1 0;
    }
    ModelsTab #model-fields,
    ModelsTab #models-transcription-fields,
    ModelsTab #models-language-fields {
        height: auto;
    }
    ModelsTab .speaker-state {
        height: auto;
        margin: 0 0 1 0;
    }
    ModelsTab .speaker-state .field-label {
        width: 18;
        text-style: bold;
    }
    ModelsTab .speaker-state .speaker-status {
        width: 26;
        height: 1;
    }
    ModelsTab .speaker-state Button {
        width: 12;
    }
    """

    class Loaded(Message):
        def __init__(self, states: list[AssetState]) -> None:
            super().__init__()
            self.states = states

    class Started(Message):
        def __init__(self, asset_id: str) -> None:
            super().__init__()
            self.asset_id = asset_id

    class Progress(Message):
        def __init__(self, asset_id: str, progress: AssetProgress) -> None:
            super().__init__()
            self.asset_id = asset_id
            self.progress = progress

    class Installed(Message):
        def __init__(self, asset_id: str) -> None:
            super().__init__()
            self.asset_id = asset_id

    class Cancelled(Message):
        def __init__(self, asset_id: str) -> None:
            super().__init__()
            self.asset_id = asset_id

    class Failed(Message):
        def __init__(self, asset_id: str, error: str) -> None:
            super().__init__()
            self.asset_id = asset_id
            self.error = error

    class Removed(Message):
        def __init__(self, asset_id: str, freed: int) -> None:
            super().__init__()
            self.asset_id = asset_id
            self.freed = freed

    def __init__(self, store: ModelAdminStore, raw: dict, infra, **kwargs) -> None:
        super().__init__(raw, **kwargs)
        self._store = store
        self._infra = infra
        self._states: dict[str, AssetState] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._chooser = ModelChooser(
            [asset for asset in store.catalog() if asset.group == 'transcription'],
            str(raw.get('transcription', {}).get('model', '')),
        )
        self._language = LanguageModelsEditor(
            raw.get('transcription', {}).get('models', {}),
            [asset for asset in store.catalog() if asset.group == 'transcription'],
        )
        self._speaker_status = Static('Checking...', classes='speaker-status')
        self._speaker_install = Button('Install', id='speaker-install')
        self._speaker_remove = Button('Remove', id='speaker-remove')

    def compose(self) -> ComposeResult:
        digest = self._raw.get('digest', {})
        interactive = self._raw.get('interactive', {})
        transcription = self._raw.get('transcription', {})

        yield section_title('Text generation')
        yield Rule(classes='section-rule')
        yield SelectRow(
            'Provider',
            'cfg-llm-provider',
            discover_providers(LLM_PROVIDERS_GROUP, BUILTIN_LLM_PROVIDERS),
            self._infra.llm_provider,
            help_text='Ollama is local. OpenAI uses a cloud API.',
        )
        yield FieldRow('Server', 'cfg-ollama-host', self._infra.ollama.host, placeholder='http://localhost:11434')
        yield FieldRow(
            'API base', 'cfg-openai-base-url', self._infra.openai.base_url, placeholder='https://api.openai.com/v1'
        )
        yield FieldRow(
            'API key',
            'cfg-openai-api-key',
            self._infra.openai.api_key or '',
            password=True,
            placeholder='sk-...',
        )
        yield Button('Clear stored key', id='cfg-openai-clear-key')
        with Vertical(id='model-fields'):
            yield FieldRow('Summary model', 'cfg-digest-model', str(digest.get('model', '')))
            yield FieldRow('Quick action model', 'cfg-interactive-model', str(interactive.get('model', '')))
        yield Static(
            'Models are managed by the plugin. Edit config.yaml to configure.',
            id='plugin-model-note',
            classes='section-desc',
        )
        yield Static(
            'Press Refresh to list the models on the server.',
            id='cfg-ollama-models',
            classes='section-desc',
        )
        with Horizontal(classes='button-row'):
            yield Button('Test', id='cfg-ai-test')
            yield Button('Refresh', id='cfg-ollama-refresh')
            yield Button('Pull summary', id='cfg-ollama-pull-digest')
            yield Button('Pull quick action', id='cfg-ollama-pull-interactive')

        yield section_title('Speech-to-Text')
        yield Rule(classes='section-rule')
        yield SelectRow(
            'Engine',
            'cfg-trans-provider',
            discover_providers(TRANSCRIPTION_PROVIDERS_GROUP, BUILTIN_TRANSCRIPTION_PROVIDERS),
            self._infra.transcription_provider,
            help_text='whisper-cpp runs locally. Plugin engines appear when installed.',
        )
        yield Static(
            'Models are managed by the engine plugin. Edit config.yaml to configure.',
            id='plugin-trans-model-note',
            classes='section-desc',
        )
        with Vertical(id='models-transcription-fields'):
            yield Static('Default model', classes='subsection-title')
            yield self._chooser
            with Horizontal(classes='button-row'):
                yield Button('Download all missing', id='models-download-all')

        with Vertical(id='models-language-fields'):
            yield section_title('Language models')
            yield Rule(classes='section-rule')
            yield Static(
                'Use a different model for a locale. The default applies otherwise.',
                classes='section-desc',
            )
            yield self._language

        yield section_title('Speaker assets')
        yield Rule(classes='section-rule')
        with Horizontal(classes='speaker-state'):
            yield Static('Runtime and model', classes='field-label')
            yield self._speaker_status
            yield self._speaker_install
            yield self._speaker_remove

        yield section_title('Advanced')
        yield Rule(classes='section-rule')
        custom = str(transcription.get('model', ''))
        yield Input(
            value=custom if Path(custom).is_absolute() else '',
            id='cfg-trans-custom-path',
            placeholder='Absolute path to a custom GGML file',
        )
        yield Static(f'Cache: {self._store.cache_root}', classes='section-desc')
        yield Button('Open cache folder', id='models-open-cache')

    def on_mount(self) -> None:
        self.run_worker(self._load_states, thread=True, group='models-load')

    def _load_states(self) -> None:
        self.post_message(self.Loaded(self._store.list_assets()))

    def model_value(self) -> str:
        return self._chooser.selected_uri or str(self._raw.get('transcription', {}).get('model', ''))

    def language_models(self) -> dict[str, str]:
        return self._language.values()

    def reload(self, raw: dict) -> None:
        self._raw = raw
        config = raw.get('transcription', {})
        self._chooser.select_by_uri(str(config.get('model', '')))
        self._language.reload(config.get('models', {}))

    # --- Messages -------------------------------------------------------

    @on(Loaded)
    def _on_loaded(self, message: Loaded) -> None:
        for state in message.states:
            self._apply_state(state)
        self._update_speaker_status()

    @on(Started)
    def _on_started(self, message: Started) -> None:
        self._row(message.asset_id).mark_downloading()

    @on(Progress)
    def _on_progress(self, message: Progress) -> None:
        self._row(message.asset_id).set_progress(message.progress)

    @on(Installed)
    def _on_installed(self, message: Installed) -> None:
        self._refresh(message.asset_id)
        self.notify(f'{self._label(message.asset_id)} is ready.', timeout=4)

    @on(Removed)
    def _on_removed(self, message: Removed) -> None:
        self._refresh(message.asset_id)
        self.notify(f'Freed {_megabytes(message.freed)}.', timeout=4)

    @on(Cancelled)
    def _on_cancelled(self, message: Cancelled) -> None:
        self._refresh(message.asset_id)
        self.notify(f'Download cancelled: {self._label(message.asset_id)}.', timeout=4)

    @on(Failed)
    def _on_failed(self, message: Failed) -> None:
        self._refresh(message.asset_id)
        self.notify(f'{self._label(message.asset_id)}: {message.error}', severity='error', timeout=10)

    # --- Actions --------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ''
        if button_id == 'models-download-all':
            self._download_all_missing()
        elif button_id == 'models-open-cache':
            self._open_cache()
        elif button_id == 'speaker-install':
            self._start_download(model_catalog.RUNTIME_ID)
            self._start_download(model_catalog.NEMOTRON_ID)
        elif button_id == 'speaker-remove':
            self._confirm_remove_speakers()
        else:
            for prefix, action in (
                ('download-', self._start_download),
                ('remove-', self._confirm_remove),
                ('cancel-', self._cancel_download),
            ):
                if button_id.startswith(prefix):
                    action(button_id.removeprefix(prefix))
                    return

    def _download_all_missing(self) -> None:
        for asset in self._store.catalog():
            state = self._states.get(asset.id)
            if state is None or state.status in (AssetStatus.MISSING, AssetStatus.CORRUPT):
                self._start_download(asset.id)

    def _start_download(self, asset_id: str) -> None:
        if asset_id in self._cancel:
            return
        self.post_message(self.Started(asset_id))
        self.run_worker(lambda: self._install(asset_id), thread=True, group=f'model-{asset_id}')

    def _install(self, asset_id: str) -> None:
        cancel = threading.Event()
        self._cancel[asset_id] = cancel

        def report(progress: AssetProgress) -> None:
            self.post_message(self.Progress(asset_id, progress))

        try:
            self._store.install(asset_id, on_progress=report, cancel=cancel)
        except DownloadCancelledError:
            self.post_message(self.Cancelled(asset_id))
        except Exception as exc:  # noqa: BLE001 -- surface every install failure in the row
            self.post_message(self.Failed(asset_id, str(exc)))
        else:
            self.post_message(self.Installed(asset_id))
        finally:
            self._cancel.pop(asset_id, None)

    def _cancel_download(self, asset_id: str) -> None:
        cancel = self._cancel.get(asset_id)
        if cancel is not None:
            cancel.set()

    def _confirm_remove(self, asset_id: str) -> None:
        asset = self._asset(asset_id)
        self.app.push_screen(
            RemoveModelModal(asset.label, asset.size),
            callback=lambda confirmed: self._remove(asset_id) if confirmed else None,
        )

    def _confirm_remove_speakers(self) -> None:
        assets = [self._asset(model_catalog.RUNTIME_ID), self._asset(model_catalog.NEMOTRON_ID)]
        size = sum(asset.size for asset in assets)
        self.app.push_screen(
            RemoveModelModal('Speaker runtime and model', size),
            callback=lambda confirmed: self._remove_speakers() if confirmed else None,
        )

    def _remove_speakers(self) -> None:
        for asset_id in (model_catalog.RUNTIME_ID, model_catalog.NEMOTRON_ID):
            self._remove(asset_id)

    def _remove(self, asset_id: str) -> None:
        self.run_worker(lambda: self._remove_worker(asset_id), thread=True, group=f'remove-{asset_id}')

    def _remove_worker(self, asset_id: str) -> None:
        try:
            freed = self._store.remove(asset_id)
        except Exception as exc:  # noqa: BLE001 -- surface every removal failure in the row
            self.post_message(self.Failed(asset_id, str(exc)))
        else:
            self.post_message(self.Removed(asset_id, freed))

    def _open_cache(self) -> None:
        opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
        subprocess.Popen([opener, str(self._store.cache_root)])  # noqa: S603 -- fixed arg list, not shell=True

    # --- Helpers --------------------------------------------------------

    def _apply_state(self, state: AssetState) -> None:
        self._states[state.asset.id] = state
        if state.asset.group == 'transcription':
            self._chooser.set_state(state)

    def _refresh(self, asset_id: str) -> None:
        self._apply_state(self._store.status(asset_id))
        self._update_speaker_status()

    def _update_speaker_status(self) -> None:
        runtime = self._states.get(model_catalog.RUNTIME_ID)
        model = self._states.get(model_catalog.NEMOTRON_ID)
        if runtime is None or model is None:
            text = 'Checking...'
            ready = False
        elif runtime.status is AssetStatus.READY and model.status is AssetStatus.READY:
            text = f'Ready ({_megabytes(runtime.asset.size + model.asset.size)})'
            ready = True
        else:
            text = 'Missing'
            ready = False
        self._speaker_status.update(text)
        self._speaker_install.display = not ready
        self._speaker_remove.display = ready

    def _row(self, asset_id: str) -> ModelChoiceRow:
        return self.query_one(f'#choice-{asset_id}', ModelChoiceRow)

    def _asset(self, asset_id: str):
        return next(asset for asset in self._store.catalog() if asset.id == asset_id)

    def _label(self, asset_id: str) -> str:
        return self._asset(asset_id).label
