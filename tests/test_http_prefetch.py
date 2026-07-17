from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from my_sd.data.tar_stream import AsyncShardPrefetcher


def test_http_prefetch_resumes_existing_part_file(tmp_path) -> None:
    payload = bytes(range(256)) * 4096
    requests: list[str | None] = []

    class RangeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_value = self.headers.get("Range")
            requests.append(range_value)
            start = 0
            status = 200
            if range_value:
                start = int(range_value.removeprefix("bytes=").removesuffix("-"))
                status = 206
            body = payload[start:]
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            if status == 206:
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/sample.tar"
        prefetcher = AsyncShardPrefetcher(
            [url],
            tmp_path / "cache",
            prefetch=1,
            delete_after_use=False,
        )
        destination = prefetcher.cache_dir / (
            f"{__import__('hashlib').sha1(url.encode()).hexdigest()[:16]}-sample.tar"
        )
        temporary = destination.with_suffix(".tar.part")
        temporary.write_bytes(payload[:12345])
        paths = list(prefetcher)
        assert paths[0].read_bytes() == payload
        assert requests == ["bytes=12345-"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
