"""Run with python -m chemgraph.api; this service uses one API process."""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
        from chemgraph.api.app import create_app
    except ImportError as exc:
        parser.error(f"Install chemgraph[web] to run the HTTP API: {exc}")
    uvicorn.run(create_app(), host=args.host, port=args.port, ws="none")


if __name__ == "__main__":
    main()
