from __future__ import annotations

import socket
import asyncio
import ipaddress
from typing import Protocol
from collections.abc import Mapping

import ifaddr
from zeroconf import ServiceInfo, ServiceStateChange
from zeroconf.asyncio import AsyncZeroconf, AsyncServiceInfo, AsyncServiceBrowser

_SERVICE_TYPE = "_lumivox-voice._tcp.local."
_PROTOCOL = b"lumivox.voice.v1"


class _ServiceBrowser(Protocol):
    @property
    def revision(self) -> int: ...

    async def resolve(self, timeout_s: float) -> tuple[str, ...]: ...

    async def wait_for_update(self, revision: int) -> None: ...

    async def close(self) -> None: ...


class _ServiceAdvertiser(Protocol):
    async def close(self) -> None: ...


class _ZeroconfBrowser:
    def __init__(self, service_id: str) -> None:
        self._name = f"{service_id}.{_SERVICE_TYPE}"
        self._zeroconf = AsyncZeroconf()
        self._candidates: tuple[str, ...] = ()
        self._revision = 0
        self._updated = asyncio.Condition()
        self._tasks: set[asyncio.Task[None]] = set()
        self._event_generation = 0
        self._closed = False
        self._browser = AsyncServiceBrowser(
            self._zeroconf.zeroconf,
            _SERVICE_TYPE,
            handlers=[self._service_changed],
        )

    @property
    def revision(self) -> int:
        return self._revision

    async def resolve(self, timeout_s: float) -> tuple[str, ...]:
        async with asyncio.timeout(timeout_s):
            async with self._updated:
                await self._updated.wait_for(lambda: bool(self._candidates) or self._closed)
                if self._closed:
                    raise RuntimeError("service browser is closed")
                return self._candidates

    async def wait_for_update(self, revision: int) -> None:
        async with self._updated:
            await self._updated.wait_for(lambda: self._revision != revision or self._closed)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._browser.async_cancel()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._updated:
            self._updated.notify_all()
        await self._zeroconf.async_close()

    def _service_changed(
        self,
        _zeroconf: object,
        service_type: str,
        name: str,
        state: ServiceStateChange,
    ) -> None:
        if self._closed or service_type != _SERVICE_TYPE or name != self._name:
            return
        self._event_generation += 1
        generation = self._event_generation
        if state is ServiceStateChange.Removed:
            task = asyncio.create_task(self._replace_candidates((), generation), name="linklab-discovery-remove")
        else:
            task = asyncio.create_task(self._refresh(generation), name="linklab-discovery-refresh")
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _refresh(self, generation: int) -> None:
        info = AsyncServiceInfo(_SERVICE_TYPE, self._name)
        if not await info.async_request(self._zeroconf.zeroconf, 3_000):
            return
        properties = info.properties
        candidates = _service_candidates(properties, info.parsed_scoped_addresses(), info.port)
        await self._replace_candidates(candidates, generation)

    async def _replace_candidates(self, candidates: tuple[str, ...], generation: int) -> None:
        async with self._updated:
            if generation != self._event_generation:
                return
            self._candidates = candidates
            self._revision += 1
            self._updated.notify_all()


class _ZeroconfAdvertiser:
    def __init__(self, zeroconf: AsyncZeroconf, info: ServiceInfo) -> None:
        self._zeroconf = zeroconf
        self._info = info
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._zeroconf.async_unregister_service(self._info)
        finally:
            await self._zeroconf.async_close()


async def _open_service_browser(service_id: str) -> _ServiceBrowser:
    return _ZeroconfBrowser(service_id)


async def _register_service(
    service_id: str,
    host: str,
    port: int,
    *,
    secure: bool,
) -> _ServiceAdvertiser:
    addresses = await _advertised_addresses(host)
    if not addresses:
        raise OSError("discovery requires at least one usable non-loopback address")
    zeroconf = AsyncZeroconf()
    info = ServiceInfo(
        _SERVICE_TYPE,
        f"{service_id}.{_SERVICE_TYPE}",
        addresses=[address.packed for address in addresses],
        port=port,
        properties={"protocol": _PROTOCOL, "scheme": b"wss" if secure else b"ws"},
        server=_local_server_name(),
    )
    try:
        await zeroconf.async_register_service(info, allow_name_change=False)
    except BaseException:
        await zeroconf.async_close()
        raise
    return _ZeroconfAdvertiser(zeroconf, info)


async def _advertised_addresses(host: str) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    if host in ("", "0.0.0.0", "::"):
        raw = [
            ip.ip[0] if isinstance(ip.ip, tuple) else ip.ip for adapter in ifaddr.get_adapters() for ip in adapter.ips
        ]
    else:
        loop = asyncio.get_running_loop()
        resolved = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        raw = [item[4][0] for item in resolved]
    addresses = (ipaddress.ip_address(value.split("%", 1)[0]) for value in raw)
    return tuple(dict.fromkeys(address for address in addresses if _usable_ip(address)))


def _usable_address(value: str) -> bool:
    try:
        return _usable_ip(ipaddress.ip_address(value.split("%", 1)[0]))
    except ValueError:
        return False


def _usable_ip(value: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (value.is_loopback or value.is_unspecified or value.is_multicast)


def _endpoint_uri(scheme: str, address: str, port: int) -> str:
    host = f"[{address.replace('%', '%25')}]" if ":" in address else address
    return f"{scheme}://{host}:{port}"


def _service_candidates(
    properties: Mapping[bytes, bytes | None],
    addresses: list[str],
    port: int | None,
) -> tuple[str, ...]:
    protocol = properties.get(b"protocol")
    scheme = properties.get(b"scheme")
    if protocol != _PROTOCOL or scheme not in (b"ws", b"wss") or not port:
        return ()
    return tuple(
        dict.fromkeys(
            _endpoint_uri(scheme.decode("ascii"), address, port) for address in addresses if _usable_address(address)
        )
    )


def _local_server_name() -> str:
    hostname = socket.getfqdn().rstrip(".")
    if "." not in hostname:
        hostname = f"{hostname}.local"
    return f"{hostname}."
