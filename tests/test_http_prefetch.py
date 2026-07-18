from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from my_sd.data.tar_stream import (
    AsyncShardPrefetcher,
    _download_headers,
    _parse_hf_source,
    shard_download_options,
)


def test_download_options_convert_gibibytes_to_bytes() -> None:
    options = shard_download_options(
        {
            "download_retries": 7,
            "download_timeout_seconds": 45,
            "minimum_free_gb": 1.5,
            "max_cache_gb": 3,
        }
    )
    assert options == {
        "download_retries": 7,
        "download_timeout_seconds": 45,
        "minimum_free_bytes": int(1.5 * 1024**3),
        "max_cache_bytes": 3 * 1024**3,
    }


def test_huggingface_headers_use_environment_token(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    headers = _download_headers(
        "https://huggingface.co/datasets/owner/repo/resolve/main/train/a.tar",
        existing_bytes=123,
    )
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Range"] == "bytes=123-"


def test_huggingface_headers_use_cached_login(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        "huggingface_hub.get_token",
        lambda: "cached-token",
    )
    headers = _download_headers(
        "https://huggingface.co/datasets/owner/repo/resolve/main/train/a.tar"
    )
    assert headers["Authorization"] == "Bearer cached-token"


def test_parse_hf_dataset_source() -> None:
    assert _parse_hf_source(
        "hf://datasets/deepghs/danbooru2024-webp-4Mpixel/images/0000.tar"
    ) == (
        "deepghs/danbooru2024-webp-4Mpixel",
        "images/0000.tar",
        "dataset",
    )
    assert _parse_hf_source("https://example.test/0000.tar") is None


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
