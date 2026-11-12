"""Operator CLI; service and scientific dependencies are imported only on use."""

import argparse
import asyncio
import json
import logging


def add_service_args(parser):
    parser.add_argument("--config", dest="service_config", required=True, help="Service TOML configuration.")
    commands = parser.add_subparsers(dest="service_action", required=True)
    commands.add_parser("serve", help="Run the authenticated single-process HTTP service.")
    tokens = commands.add_parser("token", help="Issue and manage local bearer tokens.")
    token_commands = tokens.add_subparsers(dest="token_action", required=True)
    issue = token_commands.add_parser("issue")
    issue.add_argument("user", help="Stable operator-managed user name.")
    issue.add_argument("--days", type=int, default=30)
    rotate = token_commands.add_parser("rotate")
    rotate.add_argument("token_id")
    rotate.add_argument("--days", type=int, default=30)
    revoke = token_commands.add_parser("revoke")
    revoke.add_argument("token_id")
    token_commands.add_parser("list")
    jobs = commands.add_parser("job", help="Reconcile uncertain operations with operator evidence.")
    job_commands = jobs.add_subparsers(dest="job_action", required=True)
    abandon = job_commands.add_parser("abandon")
    abandon.add_argument("job_id")
    abandon.add_argument("--reason", required=True)
    confirm = job_commands.add_parser("confirm-nonexecution")
    confirm.add_argument("job_id")
    confirm.add_argument("--evidence", required=True)
    approve = job_commands.add_parser("approve")
    approve.add_argument("job_id")


def run_service(args):
    from chemgraph.mcp.service.config import load_config

    config = load_config(args.service_config)
    if args.service_action == "serve":
        import uvicorn
        from chemgraph.mcp.service.app import create_app

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        uvicorn.run(create_app(config), host=config.host, port=config.port, ws="none",
                    proxy_headers=False, workers=1)
    elif args.service_action == "token":
        from chemgraph.mcp.service.auth import TokenStore

        tokens = TokenStore(config.database)
        if args.token_action == "issue":
            print(tokens.issue(args.user, lifetime=args.days * 86400))
        elif args.token_action == "rotate":
            print(tokens.rotate(args.token_id, lifetime=args.days * 86400))
        elif args.token_action == "revoke":
            print(json.dumps({"revoked": tokens.revoke(args.token_id)}))
        else:
            print(json.dumps(tokens.list_tokens(), indent=2))
    else:
        from chemgraph.runtime import OperationJournal, SqliteOperationStore
        from chemgraph.runtime.policy import validate, PreparedAction
        from chemgraph.mcp.service.operations import Service, call_from_record

        async def reconcile():
            journal = OperationJournal(SqliteOperationStore(config.database))
            if args.job_action == "abandon":
                return await journal.abandon(args.job_id, decided_by="operator", reason=args.reason)
            if args.job_action == "confirm-nonexecution":
                return await journal.confirm_nonexecution(args.job_id, decided_by="operator", evidence=args.evidence)
            record = await journal.get(args.job_id)
            if record is None or record["prepared"] is None:
                raise ValueError("Unknown prepared operation.")
            service = Service(config)
            await asyncio.to_thread(validate, service.record_context(record), call_from_record(record),
                                    PreparedAction.from_dict(record["prepared"]))
            return await journal.decide(record["id"], digest=record["prepared"]["digest"],
                                        approve=True, decided_by="operator")
        result = asyncio.run(reconcile())
        print(json.dumps({"batch_id": result["id"], "status": result["state"]}))


def main():
    parser = argparse.ArgumentParser(description="ChemGraph persistent MCP service")
    add_service_args(parser)
    run_service(parser.parse_args())
