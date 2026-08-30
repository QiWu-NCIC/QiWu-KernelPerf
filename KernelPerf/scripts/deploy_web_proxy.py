from __future__ import annotations

import argparse
import shlex
import subprocess

from common import PROJECT_ROOT, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the configured KernelPerf web proxy.")
    parser.add_argument("--deployment", default="config/deployment.json")
    args = parser.parse_args()
    proxy = load_json(args.deployment)["web_proxy"]
    target = str(proxy["ssh_target"])
    remote_script = "/tmp/kernelperf_web_proxy.py"
    subprocess.run(
        ["scp", str(PROJECT_ROOT / "scripts/kernelperf_web_proxy.py"), f"{target}:{remote_script}"],
        check=True,
    )
    values = {key: shlex.quote(str(value)) for key, value in proxy.items()}
    command = (
        f"docker network inspect {values['network']} >/dev/null 2>&1 || docker network create {values['network']}; "
        f"docker network connect {values['network']} {values['application_container']} >/dev/null 2>&1 || true; "
        f"docker rm -f {values['container']} >/dev/null 2>&1 || true; "
        f"docker run -d --name {values['container']} --restart unless-stopped "
        f"--network {values['network']} -p {values['listen_port']}:{values['listen_port']} "
        f"-v {shlex.quote(remote_script)}:/opt/kernelperf_web_proxy.py:ro "
        f"--entrypoint /usr/bin/python3 {values['image']} /opt/kernelperf_web_proxy.py "
        f"--listen-host {values['listen_host']} --listen-port {values['listen_port']} "
        f"--target-host {values['target_host']} --target-port {values['target_port']}"
    )
    subprocess.run(["ssh", target, command], check=True)


if __name__ == "__main__":
    main()
