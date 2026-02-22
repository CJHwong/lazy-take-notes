"""App subclasses — BaseApp, RecordApp, TranscribeApp, ViewApp."""

from lazy_take_notes.l4_frameworks_and_drivers.apps.base import BaseApp
from lazy_take_notes.l4_frameworks_and_drivers.apps.record import RecordApp
from lazy_take_notes.l4_frameworks_and_drivers.apps.transcribe import TranscribeApp
from lazy_take_notes.l4_frameworks_and_drivers.apps.view import ViewApp

__all__ = ['BaseApp', 'RecordApp', 'TranscribeApp', 'ViewApp']
