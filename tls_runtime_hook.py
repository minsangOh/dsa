"""PyInstaller runtime hook to configure TLS for LibreSSL builds."""
from ssl_compat import configure_tls

configure_tls()
