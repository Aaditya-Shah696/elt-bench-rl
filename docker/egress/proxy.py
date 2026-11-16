"""egress proxy, the only way out of the sandbox network.

sandboxes have no internet. for a cloud warehouse they set HTTPS_PROXY to this, and it only
lets https through to hosts on the allow list (snowflake + its stage buckets). everything else,
like huggingface where the answers live, gets a 403
"""

import asyncio
import os
import re

ALLOW = [re.compile(p) for p in os.environ.get("ALLOW", "").split(",") if p]
PORT = int(os.environ.get("PORT", "3128"))


def allowed(host: str, port: int) -> bool:
    return port == 443 and any(p.search(host) for p in ALLOW)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        method, target, _ = head.split(b"\r\n", 1)[0].decode().split(" ", 2)
        host, _, port = target.rpartition(":")
        if method != "CONNECT" or not allowed(host, int(port)):
            print(f"deny  {method} {target}", flush=True)
            writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\nblocked by elt-egress\r\n")
            await writer.drain()
            writer.close()
            return
        up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 30)
        print(f"allow {target}", flush=True)
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
    except Exception as e:  # noqa: BLE001 - one bad client shouldn't take the proxy down
        print(f"error {type(e).__name__}: {e}", flush=True)
        writer.close()


async def main() -> None:
    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    print(f"egress proxy on :{PORT}, allowing {[p.pattern for p in ALLOW]}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
