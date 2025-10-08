"""Utilities to improve TLS compatibility on macOS LibreSSL builds."""
from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path
from typing import Iterable

try:
    import certifi
except ImportError:
    certifi = None


def _setenv_if_missing(pairs: Iterable[tuple[str, str]]) -> None:
    """Populate environment variables when they are unset.""" 
    for key, value in pairs:
        if key not in os.environ or not os.environ[key]:
            os.environ[key] = value


def configure_tls() -> None:
    """Force requests/urllib3 to rely on the certifi CA bundle.

    The system Python shipped with recent macOS releases may be linked against
    LibreSSL, which ships with a minimal CA set. When the bundle is incomplete
    HTTPS handshakes can fail. We align the runtime with certifi's curated
    certificate store and attempt to prime the default SSL context so that both
    ``requests`` and lower-level ``ssl`` consumers honour the same bundle.
    """
    if not certifi:
        print("Warning: certifi is not installed. SSL verification may fail.", file=sys.stderr)
        return

    cafile = certifi.where()
    cadir = str(Path(cafile).resolve().parent)

    _setenv_if_missing([
        ("SSL_CERT_FILE", cafile),
        ("SSL_CERT_DIR", cadir),
        ("REQUESTS_CA_BUNDLE", cafile),
    ])

    try:
        context = ssl.create_default_context()
        context.load_verify_locations(cafile)
    except ssl.SSLError:
        # Fallback on macOS LibreSSL builds where the default context is not
        # yet initialised with certifi's bundle. Ignored because we already
        # exported the required environment variables.
        pass

    try:
        ssl._create_default_https_context = ssl.create_default_context  # type: ignore[attr-defined]
    except AttributeError:
        pass

    try:
        import urllib3.contrib.securetransport  # pylint: disable=import-outside-toplevel

        urllib3.contrib.securetransport.inject_into_urllib3()
    except Exception:
        # securetransport is only available on macOS; safely ignore if missing.
        pass


__all__ = ["configure_tls"]
