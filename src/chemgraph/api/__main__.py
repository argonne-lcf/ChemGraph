"""Run with python -m chemgraph.api; this service uses one API process."""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration without contacting providers or starting workers.",
    )
    args = parser.parse_args()
    from chemgraph.api.settings import ConfigurationError, Settings
    from chemgraph.api.providers import model_status

    try:
        settings = Settings.from_env()
        readiness = model_status(settings)
    except ConfigurationError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    if args.check_config:
        import json

        print(
            json.dumps(
                {"model_status": readiness, "connectivity_verified": False}, indent=2
            )
        )
        parser.exit(
            0
            if readiness and all(item["configured"] for item in readiness.values())
            else 1
        )
    try:
        import uvicorn
        from chemgraph.api.app import create_app
    except ImportError as exc:
        parser.error(f"Install chemgraph[web] to run the HTTP API: {exc}")
    uvicorn.run(create_app(settings), host=args.host, port=args.port, ws="none")


if __name__ == "__main__":
    main()
