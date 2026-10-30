"""Offline inspection of existing scoped-run artifact bytes."""

import json
import sys

from chemgraph.runtime.artifact_audit import audit_journal


def add_artifact_args(parser):
    commands = parser.add_subparsers(dest="artifact_command", required=True)
    verify = commands.add_parser(
        "verify", help="Check recorded output bytes and optional input snapshots.",
        epilog=("Exit codes: 0 = all requested bytes match (at least one descriptor); "
                "1 = audit issues/incomplete read; 2 = invalid arguments/roots; "
                "3 = no descriptors. Non-finalized records are skipped. "
                "Matching bytes do not establish execution success or scientific acceptance."),
    )
    for option in ("workspace", "journal", "thread-id"):
        verify.add_argument(f"--{option}", required=True)
    verify.add_argument("--include-inputs", action="store_true", help="Also check prepared input snapshots.")
    verify.add_argument("--json", action="store_true", help="Print one JSON audit report.")


def run_artifacts(args):
    try:
        report = audit_journal(workspace=args.workspace, journal=args.journal,
                               thread_id=args.thread_id, include_inputs=args.include_inputs)
    except (OSError, ValueError) as exc:
        print(f"Artifact audit: {str(exc)!r}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, allow_nan=False))
    else:
        print(f"Artifact audit: {report['outcome']} (thread {args.thread_id!r})")
        for item in report["records"]:
            print(f"  {item.get('id', item['origin'])!r}: {item['status']}")
            for finding in item["descriptors"]:
                print(f"    {finding['role']}[{finding['index']}] "
                      f"{finding.get('path', '<invalid>')!r}: {finding['status']}")
        for issue in report["issues"]:
            print(f"  {issue['origin']!r}: {issue['status']} ({issue['detail']!r})")
        counts = report["counts"]
        print(f"Checked {counts['checked']}/{counts['requested']} descriptors; "
              f"skipped {counts['skipped']} non-finalized records.")
        print("Bytes are compared to local descriptors. Execution outcome: unknown; "
              "scientific acceptance: unassessed.")
    return {"verified": 0, "issues": 1, "no_descriptors": 3}[report["outcome"]]
