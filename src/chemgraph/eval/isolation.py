"""Fail-closed probes executed inside the prepared worker before evaluation."""

import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import sys
import time
from urllib.parse import urlsplit

HOSTS = ("apps.inside.anl.gov", "pubchem.ncbi.nlm.nih.gov")


def proxy_request(proxy, method, target):
    with socket.create_connection((proxy, 3128), timeout=10) as sock:
        sock.sendall(f"{method} {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        header = b""
        while b"\r\n\r\n" not in header:
            part = sock.recv(4096)
            if not part or len(header) > 32768:
                raise RuntimeError("Invalid proxy response")
            header += part
        status = int(header.split()[1])
        if status == 200 and method == "CONNECT":
            # Verify TLS through the tunnel without sending API requests.
            with ssl.create_default_context().wrap_socket(sock, server_hostname=target.rsplit(":", 1)[0]):
                pass
        return status


def check(cpus, memory):
    manifest = json.loads(Path("/opt/chemgraph-evaluation-manifest.json").read_text())
    for name, checksum in manifest["weights"].items():
        assert name.startswith("/opt/model-cache/")
        with open(name, "rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == checksum, "Weight checksum mismatch"
    assert os.getuid() != 0, "Worker must not be root"
    status = Path("/proc/self/status").read_text()
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    assert int(fields["CapEff"].strip(), 16) == 0
    assert fields["NoNewPrivs"].strip() == "1"
    assert fields["Seccomp"].strip() == "2"
    cgroup = Path("/sys/fs/cgroup")
    assert int((cgroup / "memory.max").read_text()) == memory
    assert int((cgroup / "pids.max").read_text()) == 256
    quota, period = (cgroup / "cpu.max").read_text().split()
    assert abs(int(quota) / int(period) - cpus) < 0.001
    try:
        Path("/etc/chemgraph-write-probe").touch()
    except OSError:
        pass
    else:
        raise AssertionError("Root filesystem is writable")
    assert os.statvfs("/").f_flag & os.ST_RDONLY, "Root filesystem is not read-only"
    assert int((cgroup / "memory.swap.max").read_text()) == 0
    assert os.access("/workspace", os.W_OK)
    assert not any(Path("/workspace").iterdir()), "Workspace is not private and empty"
    for route in Path("/proc/net/route").read_text().splitlines()[1:]:
        assert route.split()[1] != "00000000", "Worker has a default IPv4 route"
    for route in Path("/proc/net/ipv6_route").read_text().splitlines():
        parts = route.split()
        # Kernel unreachable defaults have RTF_REJECT set; they cannot route traffic.
        assert not (parts[0] == "0" * 32 and parts[1] == "00" and not int(parts[8], 16) & 0x200)
    for host in (*HOSTS, "example.com", "host.containers.internal"):
        try:
            socket.getaddrinfo(host, 443)
        except socket.gaierror:
            pass
        else:
            raise AssertionError(f"External DNS/host alias resolved: {host}")
    for host in ("1.1.1.1", "8.8.8.8", "169.254.169.254"):
        try:
            connection = socket.create_connection((host, 443), timeout=2)
        except OSError:
            pass
        else:
            connection.close()
            raise AssertionError(f"Direct egress available: {host}")
    proxy = urlsplit(os.environ["HTTPS_PROXY"]).hostname
    for _ in range(20):
        try:
            with socket.create_connection((proxy, 3128), timeout=1):
                break
        except OSError:
            time.sleep(0.25)
    for method, target in [
        ("CONNECT", "example.com:443"), ("CONNECT", "1.1.1.1:443"),
        ("CONNECT", "[::1]:443"), ("CONNECT", HOSTS[0] + ":80"),
        ("CONNECT", "sub." + HOSTS[0] + ":443"),
        ("GET", "cache_object://localhost/info"), ("GET", "http://" + HOSTS[0] + "/"),
    ]:
        assert proxy_request(proxy, method, target) == 403, f"Proxy allowed {method} {target}"
    for host in HOSTS:
        assert proxy_request(proxy, "CONNECT", host + ":443") == 200, f"Cannot reach {host} with verified TLS"
    print(json.dumps({"isolation": "passed"}))


if __name__ == "__main__":
    check(float(sys.argv[1]), int(sys.argv[2]))
