"""Product entry point."""

from __future__ import annotations

import uvicorn

from .app import create_app
from .settings import GatewaySettings


def main() -> None:
    settings = GatewaySettings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=settings.bind_port,
        workers=1,
        reload=False,
        proxy_headers=False,
        server_header=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
