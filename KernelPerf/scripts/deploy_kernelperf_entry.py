from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from common import PROJECT_ROOT, load_json


SOURCE_DIRECTORIES = ("benchmarks", "config", "kernelperf", "scripts", "web")
SOURCE_FILES = (
    "pyproject.toml",
    "README.md",
    "DEPLOY.md",
    "doc/ARCH.md",
    "doc/API_AND_BENCHMARKS.md",
)


def _archive(path: Path) -> None:
    def include(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    with tarfile.open(path, "w:gz") as archive:
        for name in (*SOURCE_DIRECTORIES, *SOURCE_FILES):
            archive.add(PROJECT_ROOT / name, arcname=name, filter=include)


def _ssh(target: str, command: str, *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", target, command],
        input=stdin,
        text=True,
        check=True,
        capture_output=True,
    )


def _wait_for_empty_queue(target: str, url: str, timeout: int) -> None:
    probe = (
        "import json,urllib.request; "
        f"d=json.load(urllib.request.urlopen({url!r}+'/api/v1/queue',timeout=10)); "
        "print(json.dumps(d))"
    )
    deadline = time.monotonic() + timeout
    while True:
        try:
            snapshot = json.loads(_ssh(target, f"python3 -c {shlex.quote(probe)}").stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return
        if not snapshot.get("queued_job_ids") and not snapshot.get("running"):
            return
        if time.monotonic() >= deadline:
            raise SystemExit("timed out waiting for the production queue to become empty")
        time.sleep(5)


REMOTE_INSTALL = r'''
import json, os, shutil, signal, sqlite3, sys, tarfile, time
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
stage = root.parent / (".kernelperf-stage-" + str(os.getpid()))
stage.mkdir(parents=True)
with tarfile.open(archive_path, "r:gz") as bundle:
    for member in bundle.getmembers():
        target = (stage / member.name).resolve()
        if stage.resolve() not in target.parents and target != stage.resolve():
            raise RuntimeError("unsafe archive member: " + member.name)
    bundle.extractall(stage)

service = json.loads((stage / "config/service.json").read_text())
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
backup = root / "backups" / ("refactor-" + stamp)
backup.mkdir(parents=True, exist_ok=True)
for key in ("database", "contest_database"):
    source = root / service[key]
    if source.is_file():
        destination = backup / source.name
        with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
            src.backup(dst)

deployment = json.loads((stage / "config/deployment.json").read_text())["entry"]
pid_path = root / deployment["service_pid"]
if pid_path.is_file():
    try:
        pid = int(pid_path.read_text().strip())
        environ_path = Path(f"/proc/{pid}/environ")
        if environ_path.is_file():
            secret_files = {
                b"KERNELPERF_ADMIN_TOKEN": ".kernelperf-deploy-token",
                b"KERNELPERF_GITHUB_TOKEN": ".kernelperf-github-token",
            }
            for item in environ_path.read_bytes().split(b"\0"):
                name, separator, value = item.partition(b"=")
                if separator and name in secret_files:
                    token_path = root / secret_files[name]
                    token_path.write_bytes(value)
                    token_path.chmod(0o600)
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
    except (ValueError, ProcessLookupError):
        pass

root.mkdir(parents=True, exist_ok=True)
for name in ("benchmarks", "config", "kernelperf", "scripts", "web"):
    destination = root / name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(stage / name), destination)
for name in ("pyproject.toml", "README.md", "DEPLOY.md", "doc/ARCH.md", "doc/API_AND_BENCHMARKS.md"):
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage / name), destination)
shutil.rmtree(stage)
archive_path.unlink(missing_ok=True)
print(str(backup))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the configured KernelPerf entry service.")
    parser.add_argument("--deployment", default="config/deployment.json")
    parser.add_argument("--queue-timeout", type=int, default=7200)
    args = parser.parse_args()
    entry = load_json(args.deployment)["entry"]
    target = str(entry["ssh_target"])
    install_dir = str(entry["install_dir"])
    _wait_for_empty_queue(target, str(entry["health_url"]).rstrip("/"), args.queue_timeout)

    with tempfile.TemporaryDirectory(prefix="kernelperf-deploy-") as temp_dir:
        local_archive = Path(temp_dir) / "release.tar.gz"
        _archive(local_archive)
        remote_archive = f"/tmp/kernelperf-release-{os.getpid()}.tar.gz"
        subprocess.run(["scp", str(local_archive), f"{target}:{remote_archive}"], check=True)
        installed = _ssh(
            target,
            f"python3 - {shlex.quote(install_dir)} {shlex.quote(remote_archive)}",
            stdin=REMOTE_INSTALL,
        )
        print(f"database backup: {installed.stdout.strip()}")

    python = str(entry["python"])
    _ssh(
        target,
        f"cd {shlex.quote(install_dir)} && {shlex.quote(python)} "
        "-m pip install --no-build-isolation --no-deps -e .",
    )
    admin_token = os.environ.get("KERNELPERF_ADMIN_TOKEN")
    admin_file = shlex.quote(f"{install_dir}/.kernelperf-deploy-token")
    if admin_token:
        admin_setup = (
            f"export KERNELPERF_ADMIN_TOKEN={shlex.quote(admin_token)} && "
            f"rm -f {admin_file}"
        )
    else:
        admin_setup = (
            f"test -s {admin_file} && "
            f"export KERNELPERF_ADMIN_TOKEN=\"$(cat {admin_file})\" && "
            f"rm -f {admin_file}"
        )
    github_token = os.environ.get("KERNELPERF_GITHUB_TOKEN")
    github_file = shlex.quote(f"{install_dir}/.kernelperf-github-token")
    if github_token:
        github_setup = (
            f"export KERNELPERF_GITHUB_TOKEN={shlex.quote(github_token)} && "
            f"rm -f {github_file}"
        )
    else:
        github_setup = (
            f"if test -s {github_file}; then "
            f"export KERNELPERF_GITHUB_TOKEN=\"$(cat {github_file})\"; fi; "
            f"rm -f {github_file}"
        )
    start = (
        f"cd {shlex.quote(install_dir)} && "
        f"{admin_setup} && "
        f"{github_setup} && "
        "{ "
        f"nohup {shlex.quote(python)} -m kernelperf.cli serve "
        f"--config {shlex.quote(str(entry['service_config']))} "
        f"< /dev/null > {shlex.quote(str(entry['service_log']))} 2>&1 & "
        f"echo $! > {shlex.quote(str(entry['service_pid']))}; "
        "}"
    )
    _ssh(target, start)
    health_url = str(entry["health_url"]).rstrip("/")
    probe = f"import urllib.request; print(urllib.request.urlopen({health_url!r}+'/api/v1/workers',timeout=10).status)"
    for _ in range(30):
        try:
            if _ssh(target, f"python3 -c {shlex.quote(probe)}").stdout.strip() == "200":
                print("entry deployment healthy")
                return
        except subprocess.CalledProcessError:
            pass
        time.sleep(2)
    raise SystemExit("service did not become healthy after deployment")


if __name__ == "__main__":
    main()
