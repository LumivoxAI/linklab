from typing import Never, cast

import msgpack  # type: ignore[import-untyped]

from ._errors import CodecError
from ._values import ReadableBuffer

_MAX_MESSAGE_BYTES = 262_144
_MAX_FIRST_MESSAGE_BYTES = 16_384
_MAX_DEPTH = 8
_MAX_MAP_ITEMS = 64
_MAX_ARRAY_ITEMS = 32
_MAX_STRING_BYTES = 65_536
_MAX_BINARY_BYTES = 16_384

type Primitive = None | bool | int | str | bytes | list[Primitive] | dict[str, Primitive]


class _StructuralError(Exception):
    pass


def _reject_extension(_code: int, _data: bytes) -> Never:
    raise _StructuralError("MessagePack extensions are not allowed")


def _build_map(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise _StructuralError("MessagePack map keys must be strings")
        if key in result:
            raise _StructuralError("MessagePack map keys must be unique")
        result[key] = value
    return result


def _as_bytes_view(data: ReadableBuffer) -> memoryview:
    try:
        view = memoryview(data)
    except TypeError as error:
        raise CodecError("message must be a readable buffer") from error
    if not view.contiguous:
        raise CodecError("message buffer must be contiguous")
    try:
        return view.cast("B")
    except TypeError as error:
        raise CodecError("message buffer must be byte-addressable") from error


def _validate_tree(root: object) -> dict[str, Primitive]:
    if type(root) is not dict:
        raise _StructuralError("MessagePack root must be a map")

    stack: list[tuple[object, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        if type(value) is dict:
            if depth > _MAX_DEPTH:
                raise _StructuralError("MessagePack nesting exceeds the limit")
            for child in value.values():
                if type(child) is dict or type(child) is list:
                    stack.append((child, depth + 1))
                elif child is not None and type(child) not in (bool, int, str, bytes):
                    raise _StructuralError("MessagePack value type is not allowed")
        elif type(value) is list:
            if depth > _MAX_DEPTH:
                raise _StructuralError("MessagePack nesting exceeds the limit")
            for child in value:
                if type(child) is dict or type(child) is list:
                    stack.append((child, depth + 1))
                elif child is not None and type(child) not in (bool, int, str, bytes):
                    raise _StructuralError("MessagePack value type is not allowed")
        else:
            raise _StructuralError("MessagePack value type is not allowed")

    return cast(dict[str, Primitive], root)


def _decode_primitive_message(
    data: ReadableBuffer,
    *,
    max_message_bytes: int = _MAX_MESSAGE_BYTES,
    first_message: bool = False,
) -> dict[str, Primitive]:
    """Decode one bounded MessagePack map without applying a message schema."""
    view = _as_bytes_view(data)
    effective_limit = min(max_message_bytes, _MAX_MESSAGE_BYTES)
    if first_message:
        effective_limit = min(effective_limit, _MAX_FIRST_MESSAGE_BYTES)
    if len(view) > effective_limit:
        raise CodecError(f"message exceeds the {effective_limit}-byte envelope")

    try:
        unpacked = msgpack.unpackb(
            view,
            raw=False,
            use_list=True,
            strict_map_key=True,
            object_pairs_hook=_build_map,
            ext_hook=_reject_extension,
            max_str_len=_MAX_STRING_BYTES,
            max_bin_len=_MAX_BINARY_BYTES,
            max_array_len=_MAX_ARRAY_ITEMS,
            max_map_len=_MAX_MAP_ITEMS,
            max_ext_len=0,
        )
        return _validate_tree(unpacked)
    except CodecError:
        raise
    except (ValueError, TypeError, UnicodeError, msgpack.UnpackException, _StructuralError) as error:
        raise CodecError("invalid restricted MessagePack message") from error
