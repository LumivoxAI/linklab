from typing import cast

import pytest
import msgpack  # type: ignore[import-untyped]

from lumivox_linklab import CodecError
from lumivox_linklab._codec import _decode_primitive_message


def _pack(value: object) -> bytes:
    return cast(bytes, msgpack.packb(value, use_bin_type=True))


def _nested_array(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = [value]
    return {"value": value}


def _absolute_limit_message() -> bytes:
    values = {str(index): bytes(16_384) for index in range(15)}
    remaining = 262_144 - len(_pack(values))
    for size in range(remaining, 0, -1):
        candidate = {**values, "last": bytes(size)}
        packed = _pack(candidate)
        if len(packed) == 262_144:
            return packed
    raise AssertionError("could not construct exact-limit fixture")


def test_decodes_the_complete_allowed_primitive_tree() -> None:
    packed = _pack({"array": [None, True, False, -1, 2**64 - 1, "text", b"binary", {"nested": 1}]})

    assert _decode_primitive_message(memoryview(packed)) == {
        "array": [None, True, False, -1, 2**64 - 1, "text", b"binary", {"nested": 1}]
    }


@pytest.mark.parametrize(
    "value",
    [
        {str(index): index for index in range(64)},
        {"value": list(range(32))},
        {"value": "x" * 65_536},
        {"value": bytes(16_384)},
        _nested_array(7),
    ],
)
def test_structural_limits_accept_the_exact_boundary(value: object) -> None:
    assert _decode_primitive_message(_pack(value))


@pytest.mark.parametrize(
    "value",
    [
        {str(index): index for index in range(65)},
        {"value": list(range(33))},
        {"value": "x" * 65_537},
        {"value": bytes(16_385)},
        _nested_array(8),
    ],
)
def test_structural_limits_reject_one_over(value: object) -> None:
    with pytest.raises(CodecError):
        _decode_primitive_message(_pack(value))


def test_message_envelopes_accept_exact_boundaries() -> None:
    first = _pack({"value": bytes(16_374)})
    absolute = _absolute_limit_message()
    assert len(first) == 16_384
    assert len(absolute) == 262_144

    assert _decode_primitive_message(first, first_message=True)
    assert _decode_primitive_message(absolute)


def test_connection_envelope_uses_the_supplied_limit() -> None:
    packed = _pack({"value": 1})

    assert _decode_primitive_message(packed, max_message_bytes=len(packed))
    with pytest.raises(CodecError):
        _decode_primitive_message(packed, max_message_bytes=len(packed) - 1)


@pytest.mark.parametrize(
    ("payload", "kwargs"),
    [
        (_pack({"value": bytes(16_375)}), {"first_message": True}),
        (bytes(262_145), {}),
    ],
)
def test_message_envelopes_reject_one_over_before_unpacking(
    payload: bytes,
    kwargs: dict[str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("unpackb must not be called")

    monkeypatch.setattr(msgpack, "unpackb", fail_if_called)
    with pytest.raises(CodecError):
        _decode_primitive_message(payload, **kwargs)
    assert not called


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x81\xa1x",
        _pack({"value": 1}) + _pack({"value": 2}),
        _pack([1]),
        _pack(1),
        b"\x81\x01\x02",
        _pack({b"key": 1}),
        b"\x82\xa1a\x01\xa1a\x02",
        b"\x81\xa1\xff\xc0",
        b"\x81\xa1x\xa1\xff",
        _pack({"value": 1.5}),
        _pack({"value": msgpack.ExtType(1, b"x")}),
        _pack({"value": msgpack.Timestamp(0, 0)}),
    ],
)
def test_rejects_malformed_and_forbidden_messagepack(payload: bytes) -> None:
    with pytest.raises(CodecError):
        _decode_primitive_message(payload)


def test_extreme_nesting_is_a_stable_codec_error() -> None:
    payload = b"\x81\xa1x" + b"\x91" * 1_000 + b"\x00"

    with pytest.raises(CodecError):
        _decode_primitive_message(payload)


def test_rejects_non_buffer_and_noncontiguous_input() -> None:
    with pytest.raises(CodecError):
        _decode_primitive_message("not bytes")  # type: ignore[arg-type]
    with pytest.raises(CodecError):
        _decode_primitive_message(memoryview(bytearray(4))[::2])
