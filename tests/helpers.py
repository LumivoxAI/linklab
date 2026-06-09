from __future__ import annotations

from typing import Any, Self


class NullLogger:
    def bind(self, **new_values: Any) -> Self:
        return self

    def debug(self, event: str, **kwargs: Any) -> None:
        pass

    def info(self, event: str, **kwargs: Any) -> None:
        pass

    def warning(self, event: str, **kwargs: Any) -> None:
        pass

    def error(self, event: str, **kwargs: Any) -> None:
        pass

    def critical(self, event: str, **kwargs: Any) -> None:
        pass

    def exception(self, event: str, **kwargs: Any) -> None:
        pass


NULL_LOGGER = NullLogger()


class RecordingLogger(NullLogger):
    def __init__(
        self,
        records: list[tuple[str, str, dict[str, object]]] | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        self.records = [] if records is None else records
        self.context = {} if context is None else context

    def bind(self, **new_values: Any) -> Self:
        return type(self)(self.records, {**self.context, **new_values})

    def debug(self, event: str, **kwargs: Any) -> None:
        self._record("debug", event, kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._record("info", event, kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._record("warning", event, kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._record("error", event, kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._record("critical", event, kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._record("exception", event, kwargs)

    def _record(self, level: str, event: str, fields: dict[str, object]) -> None:
        self.records.append((level, event, {**self.context, **fields}))
