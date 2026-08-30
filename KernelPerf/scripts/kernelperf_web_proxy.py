#!/usr/bin/env python3
"""Small TCP proxy used when KernelPerf runs in an un-published container."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from dataclasses import dataclass


LOGGER = logging.getLogger("kernelperf-web-proxy")
BUFFER_SIZE = 64 * 1024


@dataclass
class TargetResolver:
    host: str | None
    container: str | None
    cache_seconds: float = 5.0
    _cached_host: str | None = None
    _cached_at: float = 0.0

    async def resolve(self) -> str:
        if self.host:
            return self.host

        now = time.monotonic()
        if self._cached_host and now - self._cached_at < self.cache_seconds:
            return self._cached_host

        process = await asyncio.create_subprocess_exec(
            "/usr/bin/docker",
            "inspect",
            self.container,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"docker inspect failed: {message}")

        document = json.loads(stdout)
        networks = document[0]["NetworkSettings"]["Networks"]
        addresses = [network.get("IPAddress") for network in networks.values()]
        target = next((address for address in addresses if address), None)
        if not target:
            raise RuntimeError(f"container {self.container!r} has no IPv4 address")

        self._cached_host = target
        self._cached_at = now
        return target


async def pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(BUFFER_SIZE):
        writer.write(data)
        await writer.drain()


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (ConnectionError, BrokenPipeError):
        pass


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    resolver: TargetResolver,
    target_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        target_host = await resolver.resolve()
        upstream_reader, upstream_writer = await asyncio.open_connection(
            target_host, target_port
        )
    except Exception as exc:
        LOGGER.warning("upstream connection failed for %s: %s", peer, exc)
        await close_writer(client_writer)
        return

    tasks = {
        asyncio.create_task(pump(client_reader, upstream_writer)),
        asyncio.create_task(pump(upstream_reader, client_writer)),
    }
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except (ConnectionError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        await asyncio.gather(
            close_writer(upstream_writer),
            close_writer(client_writer),
            return_exceptions=True,
        )


async def serve(args: argparse.Namespace) -> None:
    resolver = TargetResolver(host=args.target_host, container=args.target_container)
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(
            reader, writer, resolver, args.target_port
        ),
        args.listen_host,
        args.listen_port,
        backlog=256,
    )
    targets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    target = args.target_host or f"container:{args.target_container}"
    LOGGER.info("listening on %s -> %s:%s", targets, target, args.target_port)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)

    async with server:
        await stop.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8080)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-host")
    target.add_argument("--target-container")
    parser.add_argument("--target-port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(serve(parse_args()))


if __name__ == "__main__":
    main()
