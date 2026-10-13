"""ChemGraph command line entry point with lazy optional service dispatch."""


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "service":
        try:
            from chemgraph.service.cli import main as service_main
        except ImportError as exc:
            raise SystemExit('Install service dependencies with pip install "chemgraph[service]"') from exc
        return service_main(sys.argv[2:])
    from chemgraph.cli.main import main as core_main

    return core_main()


__all__ = ["main"]
