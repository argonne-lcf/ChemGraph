#!/usr/bin/env python3
"""Build test-only images and a local TLS fixture; no runtime downloads or live API calls."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--worker-image", required=True)
parser.add_argument("--test-worker-image", default="localhost/chemgraph-eval-test:latest")
parser.add_argument("--service-image", default="localhost/chemgraph-eval-test-services:latest")
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix="chemgraph-test-images-") as directory:
    root = Path(directory)
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30",
        "-keyout", str(root / "test.key"), "-out", str(root / "test.crt"),
        "-subj", "/CN=ChemGraph offline test only",
        "-addext", "subjectAltName=DNS:apps.inside.anl.gov,DNS:pubchem.ncbi.nlm.nih.gov",
    ], check=True)
    shutil.copyfile(Path(__file__).with_name("service.py"), root / "service.py")
    # Never place the fixture's private key in the worker image, even in a discarded layer.
    (root / "Worker.Containerfile").write_text(
        f"FROM {args.worker_image}\nUSER 0\n"
        "COPY test.crt /usr/local/share/ca-certificates/chemgraph-test.crt\n"
        "RUN update-ca-certificates\n"
        "ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt\n"
        "USER 1000:1000\n"
    )
    (root / "Service.Containerfile").write_text(
        f"FROM {args.worker_image}\nCOPY service.py test.crt test.key /opt/\n"
        'ENTRYPOINT ["python", "/opt/service.py"]\n'
    )
    for recipe, tag in [("Worker.Containerfile", args.test_worker_image), ("Service.Containerfile", args.service_image)]:
        subprocess.run(["podman", "--remote=false", "build", "--pull=never", "--network=none", "--http-proxy=false",
                        "--tag", tag, "--file", str(root / recipe), directory], check=True)
