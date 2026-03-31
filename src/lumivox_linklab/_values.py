from typing import NewType
from dataclasses import dataclass
from collections.abc import Buffer

from ._enums import CoarseState, ConnectionState, ProtocolObjectKind

ConversationId = NewType("ConversationId", int)
InputId = NewType("InputId", int)
ResponseId = NewType("ResponseId", int)
OutputId = NewType("OutputId", int)

type ReadableBuffer = Buffer


@dataclass(frozen=True, slots=True)
class AnnotatedAudio:
    audio: ReadableBuffer
    generation: int
    discontinuity: bool
    speech: bool
    activated: bool
    wake_word: str | None = None


@dataclass(frozen=True, slots=True)
class ProtocolStateSnapshot:
    connection_state: ConnectionState
    conversation_id: ConversationId | None
    input_id: InputId | None
    response_id: ResponseId | None
    output_id: OutputId | None
    coarse_state: CoarseState | None
    state_revision: int | None


@dataclass(frozen=True, slots=True)
class ProtocolTombstone:
    kind: ProtocolObjectKind
    conversation_id: ConversationId | None
    input_id: InputId | None
    response_id: ResponseId | None
    output_id: OutputId | None
    terminal_state: str
