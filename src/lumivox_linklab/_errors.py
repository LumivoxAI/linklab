class LinklabError(Exception):
    """Base class for all public Linklab exceptions."""


class CodecError(LinklabError):
    """A message could not be encoded or decoded."""


class ProtocolViolation(LinklabError):
    """A message or operation violates the protocol lifecycle."""


class ConnectionClosed(LinklabError):
    """An operation requires a live connection."""


class QueueOverflow(LinklabError):
    """A bounded queue cannot accept an item."""


class WriterClosed(LinklabError):
    """An operation targets a terminal writer."""
