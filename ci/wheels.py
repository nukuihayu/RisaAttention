"""Small build/release entry point. No GPU or PyTorch is required here."""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, unquote, urldefrag, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

import yaml
from packaging.utils import parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / ".gitlab-results"
PACKAGE = {"distribution": "risa-attention", "module": "risa_attention", "extension": "_C"}
MANIFEST_START = "<!-- risa-release-manifest:v1 -->\n```json\n"
MANIFEST_END = "\n```\n<!-- /risa-release-manifest -->"


class NoRedirect(HTTPRedirectHandler):
    """Never forward private index credentials (or fall back to public PyPI)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class WheelIndex(HTMLParser):
    def __init__(self, base, names):
        super().__init__()
        self.base = base
        self.names = set(names)
        self.files = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag != "a" or "href" not in attrs:
            return
        url, fragment = urldefrag(urljoin(self.base + "/", attrs["href"]))
        name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        if name not in self.names:
            return
        digest = parse_qs(fragment).get("sha256", [""])[0]
        expected = self.base + "/packages/pypi/files/" + digest + "/" + quote(name, safe="")
        if (
            not re.fullmatch(r"[a-f0-9]{64}", digest)
            or unquote(url) != unquote(expected)
            or "data-yanked" in attrs
        ):
            raise ValueError(f"Unexpected PyPI wheel link: {name}")
        entry = {"url": expected, "sha256": digest}
        if name in self.files and self.files[name] != entry:
            raise ValueError(f"Conflicting PyPI files: {name}")
        self.files[name] = entry


def base_version():
    value = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?", value):
        raise ValueError("VERSION must be X.Y.Z, X.Y.ZrcN, or X.Y.Z.devN (no CUDA suffix)")
    if str(Version(value)) != value:
        raise ValueError("VERSION must use canonical Python version spelling")
    return value


def matrix():
    """Use the CI matrix itself as the list of expected outputs."""
    config = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8"))
    combinations = {}
    for row in config["build:wheels"]["parallel"]["matrix"]:
        axes = {key: value if isinstance(value, list) else [value] for key, value in row.items()}
        for values in itertools.product(*axes.values()):
            entry = dict(zip(axes, values, strict=True))
            group = entry["CUDA_TAG"] + "-py" + entry["PYTHON_VERSION"].replace(".", "")
            target = PACKAGE["distribution"] + "-" + group
            if target in combinations:
                raise ValueError(f"Duplicate wheel target: {target}")
            combinations[target] = {**entry, "package": PACKAGE, "group": group}
    return combinations


def selection():
    combinations = matrix()
    selected = os.environ.get("BUILD_TARGET", "all")
    targets = {
        target
        for target, entry in combinations.items()
        if selected in {"all", target, entry["group"]}
    }
    if not targets:
        raise ValueError(f"Unknown BUILD_TARGET: {selected}")
    return combinations, targets


def validate_publication(version):
    value = os.environ.get("PUBLISH", "false")
    if value not in {"false", "true"}:
        raise ValueError("PUBLISH must be false or true")
    if value == "true" and not Version(version).is_prerelease:
        on_default = os.environ.get("CI_COMMIT_BRANCH") == os.environ.get(
            "CI_DEFAULT_BRANCH", "main"
        )
        on_release_tag = os.environ.get("CI_COMMIT_TAG") == "v" + version
        if not on_default and not on_release_tag:
            raise ValueError("Stable versions can only be published from the default branch")
    return value == "true"


def sha256(path):
    with path.open("rb") as stream:
        return hash_stream(stream)


def hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def architectures(value):
    if not re.fullmatch(r"[0-9]+-real(?:;[0-9]+-real)*", value):
        raise ValueError("Set explicit RISA_CUDA_ARCHITECTURES, e.g. 80-real;120-real")
    targets = value.split(";")
    if len(set(targets)) != len(targets) or any(int(sm[:-5]) < 80 for sm in targets):
        raise ValueError("Architectures must be unique SM80+ native targets")
    return targets


def inspect_wheel(path, version, cuda, python, package):
    name, wheel_version, _, tags = parse_wheel_filename(path.name)
    abi = "cp" + python.replace(".", "")
    if name != package["distribution"] or str(wheel_version) != f"{version}+{cuda}":
        raise ValueError(f"Unexpected wheel identity: {path.name}")
    if {str(tag) for tag in tags} != {f"{abi}-{abi}-linux_x86_64"}:
        raise ValueError(f"Unexpected wheel platform/ABI: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("Wheel must contain one distribution")
        metadata = Parser().parsestr(archive.read(metadata_files[0]).decode())
        if metadata["Name"] != package["distribution"] or metadata["Version"] != str(wheel_version):
            raise ValueError("Wheel metadata and filename disagree")
        prefix = package["module"] + "/" + package["extension"] + "."
        if not any(name.startswith(prefix) and name.endswith(".so") for name in names):
            raise ValueError("Wheel is missing the compiled CUDA extension")
        runtime_version = archive.read(package["module"] + "/_version.py").decode()
        if f'__version__ = "{wheel_version}"' not in runtime_version:
            raise ValueError("Runtime version does not match wheel metadata")


def build():
    version = base_version()
    validate_publication(version)
    combinations, selected = selection()
    cuda, python = os.environ["CUDA_TAG"], os.environ["PYTHON_VERSION"]
    targets = [
        target
        for target, entry in combinations.items()
        if target in selected and entry["CUDA_TAG"] == cuda and entry["PYTHON_VERSION"] == python
    ]
    if not targets:
        return
    if sys.version_info[:2] != tuple(map(int, python.split("."))):
        raise ValueError("Active Python does not match the selected target")
    compiler = os.environ.get("CUDACXX", "nvcc")
    compiler_version = subprocess.check_output([compiler, "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", compiler_version)
    if not match or cuda != f"cu{match[1]}{match[2]}":
        raise ValueError("CUDA_TAG does not match the actual toolkit")
    sms = architectures(os.environ.get("RISA_CUDA_ARCHITECTURES", ""))
    failures = []
    for target in targets:
        try:
            build_target(target, combinations[target], version, sms)
        except Exception as error:
            # A compiler or wheel-validation failure belongs to this package.
            # Preserve successful siblings; KeyboardInterrupt/SystemExit still propagate.
            failures.append(target)
            print(f"Build failed for {target}: {error}", file=sys.stderr)
    if failures:
        raise RuntimeError("Failed package targets: " + ", ".join(failures))


def build_target(target, entry, version, sms):
    cuda, python = entry["CUDA_TAG"], entry["PYTHON_VERSION"]
    output = RESULTS / target
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    if list(output.glob("*.whl")):
        raise ValueError(f"Use a fresh output directory: {output}")
    environment = {
        **os.environ,
        "RISA_CUDA_TAG": cuda,
        "RISA_CUDA_ARCHITECTURES": ";".join(sms),
        "MAX_JOBS": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "SKBUILD_BUILD_TOOL_ARGS": "-j1",
        "NVCC_PREPEND_FLAGS": "--threads=1 --split-compile=1",
    }
    print(f"Building {target}, native targets: {sms}", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "build.py"), "--wheel", str(output)],
        env=environment,
        check=True,
    )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel, found {len(wheels)}")
    wheel = wheels[0]
    inspect_wheel(wheel, version, cuda, python, entry["package"])
    report = {
        "version": version,
        "commit": os.environ["CI_COMMIT_SHA"],
        "target": target,
        "cuda": cuda,
        "python": python,
        "architectures": sms,
        "image": entry["CUDA_IMAGE"],
        "package": entry["package"]["distribution"],
        "wheel": wheel.name,
        "sha256": sha256(wheel),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


class GitLab:
    def __init__(self):
        self.base = (
            os.environ["CI_API_V4_URL"].rstrip("/") + "/projects/" + os.environ["CI_PROJECT_ID"]
        )
        self.token = os.environ["CI_JOB_TOKEN"]

    def request(self, path, method="GET", data=None):
        headers = {"JOB-TOKEN": self.token}
        if data is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(data).encode()
        try:
            with urlopen(
                Request(self.base + path, data=data, headers=headers, method=method), timeout=120
            ) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except HTTPError as error:
            if error.code == 404 and method == "GET":
                return None
            detail = error.read().decode("utf-8", errors="replace").strip()
            suffix = f": {detail[:1000]}" if detail else ""
            raise RuntimeError(f"GitLab {method} {path}: HTTP {error.code}{suffix}") from None

    def find_tag(self, name):
        refs = [f"refs/tags/{name}", f"refs/tags/{name}^{{}}"]
        try:
            output = subprocess.check_output(
                ["git", "ls-remote", "--tags", "origin", *refs], text=True
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError("Cannot read repository tags through the Git remote") from error
        matches = {ref: sha for sha, ref in (line.split("\t", 1) for line in output.splitlines())}
        commit = matches.get(refs[1], matches.get(refs[0]))
        return {"name": name, "commit": {"id": commit}} if commit else None

    def download_job_artifacts(self, job_ids):
        RESULTS.mkdir(parents=True, exist_ok=True)
        for job_id in job_ids:
            request = Request(
                self.base + f"/jobs/{job_id}/artifacts",
                headers={"JOB-TOKEN": self.token},
            )
            try:
                with (
                    urlopen(request, timeout=300) as response,
                    tempfile.TemporaryFile() as artifact,
                ):
                    while block := response.read(1024 * 1024):
                        artifact.write(block)
                    artifact.seek(0)
                    with zipfile.ZipFile(artifact) as archive:
                        members = []
                        for member in archive.infolist():
                            parts = Path(member.filename).parts
                            if parts[:1] == (".gitlab-results",) and ".." not in parts:
                                members.append((member, parts[1:]))
                        if not members:
                            raise ValueError(f"Job {job_id} artifact has no build results")
                        for member, relative in members:
                            destination = RESULTS.joinpath(*relative)
                            if member.is_dir():
                                destination.mkdir(parents=True, exist_ok=True)
                            else:
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                with (
                                    archive.open(member) as source,
                                    destination.open("wb") as target,
                                ):
                                    shutil.copyfileobj(source, target)
            except HTTPError as error:
                raise RuntimeError(
                    f"Cannot download artifacts from job {job_id}: HTTP {error.code}"
                ) from None

    def pypi_open(self, url):
        credential = base64.b64encode(f"gitlab-ci-token:{self.token}".encode()).decode()
        return build_opener(NoRedirect()).open(
            Request(url, headers={"Authorization": "Basic " + credential}), timeout=300
        )

    def pypi_files(self, names):
        url = self.base + "/packages/pypi/simple/" + PACKAGE["distribution"]
        try:
            with self.pypi_open(url) as response:
                index = WheelIndex(self.base, names)
                index.feed(response.read().decode("utf-8"))
                return index.files
        except HTTPError as error:
            # An absent private package can redirect to PyPI if forwarding is enabled.
            # Treat that as absent locally, without following the redirect.
            location = error.headers.get("Location", "").rstrip("/")
            if error.code == 404 or (
                error.code == 302
                and location == "https://pypi.org/simple/" + PACKAGE["distribution"]
            ):
                return {}
            raise RuntimeError(f"GitLab PyPI index: HTTP {error.code}") from None

    def verify_pypi(self, report, files):
        entry = files.get(report["wheel"])
        if entry is None:
            raise ValueError(f"Wheel missing from PyPI index: {report['wheel']}")
        if entry["sha256"] != report["sha256"]:
            raise ValueError(f"Refusing to replace already uploaded content: {report['wheel']}")
        with self.pypi_open(entry["url"]) as response:
            if hash_stream(response) != report["sha256"]:
                raise ValueError(f"PyPI download checksum mismatch: {report['wheel']}")
        return entry["url"]

    def upload_pypi(self, source):
        # No --skip-existing: a duplicate must be compared by hash, never ignored.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "twine",
                "upload",
                "--non-interactive",
                "--disable-progress-bar",
                "--repository-url",
                self.base + "/packages/pypi",
                str(source),
            ],
            env={**os.environ, "TWINE_USERNAME": "gitlab-ci-token", "TWINE_PASSWORD": self.token},
            check=True,
        )


def report_valid(report, target, entry, version, commit):
    architectures(";".join(report.get("architectures", [])))
    expected = {
        "target": target,
        "cuda": entry["CUDA_TAG"],
        "python": entry["PYTHON_VERSION"],
        "image": entry["CUDA_IMAGE"],
        "version": version,
        "commit": commit,
        "package": entry["package"]["distribution"],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Build report does not belong to this release: {target}")
    if Path(report["wheel"]).name != report["wheel"] or not re.fullmatch(
        r"[a-f0-9]{64}", report["sha256"]
    ):
        raise ValueError("Invalid wheel name or checksum in report")


def previous_reports(existing, combinations, version, commit, supplied):
    if existing is None:
        return {}
    if existing.get("commit", {}).get("id") != commit:
        raise ValueError("Existing Release refers to another source commit")
    description = existing.get("description", "")
    if description.count(MANIFEST_START) != 1 or description.count(MANIFEST_END) != 1:
        raise ValueError("Malformed release manifest")
    manifest = json.loads(description.split(MANIFEST_START, 1)[1].split(MANIFEST_END, 1)[0])
    if (
        manifest.get("schema") != 1
        or manifest.get("version") != version
        or manifest.get("commit") != commit
    ):
        raise ValueError("Release manifest does not belong to this release")
    reports = manifest["reports"]
    for target, report in reports.items():
        if target not in combinations:
            raise ValueError(f"Release target missing from current matrix: {target}")
        report_valid(report, target, combinations[target], version, commit)
        if target in supplied and supplied[target] != report:
            raise ValueError(f"Refusing to change published build report: {target}")
    return reports


def release_notes(version, commit, combinations, reports, selected):
    manifest = {"schema": 1, "version": version, "commit": commit, "reports": reports}
    lines = [
        f"# RISA Attention {version}",
        "",
        f"Source commit: {commit}",
        "",
        "Linux x86_64 / Ubuntu 24.04 baseline. No GPU runtime tests run in this packaging pipeline.",
        "",
        "| Target | Result | Native SMs |",
        "| --- | --- | --- |",
    ]
    for target in combinations:
        report = reports.get(target)
        status = (
            "Available"
            if report
            else ("Failed or missing" if target in selected else "Not selected / missing")
        )
        sms = (
            ", ".join(sm.removesuffix("-real") for sm in report["architectures"]) if report else "—"
        )
        lines.append(f"| {target} | {status} | {sms} |")
    lines += [
        "",
        "Wheels are published to this project's private GitLab PyPI registry.",
        "See README.md for installation and authentication; select the CUDA suffix explicitly.",
        "Missing targets may be backfilled from the same source commit.",
        "",
        "<details><summary>Build manifest and SHA-256 checksums</summary>",
        "",
        MANIFEST_START + json.dumps(manifest, indent=2) + MANIFEST_END,
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def publish():
    version = base_version()
    do_publish = validate_publication(version)
    combinations, selected = selection()
    publish_only = os.environ.get("PUBLISH_ONLY", "false")
    if publish_only not in {"false", "true"}:
        raise ValueError("PUBLISH_ONLY must be false or true")
    artifact_jobs = os.environ.get("PUBLISH_ARTIFACT_JOBS", "")
    release_commit = os.environ.get("RELEASE_COMMIT_SHA", "")
    if publish_only == "true":
        if not do_publish:
            raise ValueError("PUBLISH_ONLY requires PUBLISH=true")
        if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", artifact_jobs):
            raise ValueError("PUBLISH_ARTIFACT_JOBS must contain comma-separated job IDs")
        if not release_commit:
            raise ValueError("RELEASE_COMMIT_SHA is required for publish-only")
        GitLab().download_job_artifacts(artifact_jobs.split(","))
    elif artifact_jobs or release_commit:
        raise ValueError("Artifact reuse variables require PUBLISH_ONLY=true")
    commit = release_commit or os.environ["CI_COMMIT_SHA"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Expected the full source commit SHA")
    reports = {}
    for target, entry in combinations.items():
        path = RESULTS / target / "result.json"
        if target in selected and path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            report_valid(report, target, entry, version, commit)
            wheel = path.parent / report["wheel"]
            inspect_wheel(wheel, version, report["cuda"], report["python"], entry["package"])
            if sha256(wheel) != report["sha256"]:
                raise ValueError(f"Wheel checksum mismatch: {target}")
            reports[target] = report
    RESULTS.mkdir(parents=True, exist_ok=True)
    notes_path = RESULTS / "release-notes.md"
    notes_path.write_text(
        release_notes(version, commit, combinations, reports, selected), encoding="utf-8"
    )
    if not reports:
        raise ValueError("No successful selected wheels; no tag or Release will be created")
    if do_publish:
        api = GitLab()
        tag_name = "v" + version
        tag = api.find_tag(tag_name)
        if tag and tag["commit"]["id"] != commit:
            raise ValueError(f"{tag_name} already refers to another commit; bump VERSION")
        release_path = "/releases/" + quote(tag_name, safe="")
        existing = api.request(release_path)
        # Once main advances, a manual pipeline on the release tag can backfill.
        # It must not create a new stable release from an arbitrary manual tag.
        if (
            not Version(version).is_prerelease
            and os.environ.get("CI_COMMIT_TAG")
            and (tag is None or existing is None)
        ):
            raise ValueError("A stable tag pipeline can only backfill an existing Release")
        supplied = reports
        reports = {
            **previous_reports(existing, combinations, version, commit, supplied),
            **supplied,
        }
        names = {report["wheel"] for report in reports.values()}
        files = api.pypi_files(names)
        # Check every conflict before uploading any new file.
        for target, report in reports.items():
            entry = files.get(report["wheel"])
            if entry is not None and entry["sha256"] != report["sha256"]:
                raise ValueError(f"Refusing to replace already uploaded content: {report['wheel']}")
            if entry is None and target not in supplied:
                raise ValueError(f"Previously released wheel missing from PyPI: {report['wheel']}")
        for target, report in supplied.items():
            if report["wheel"] not in files:
                api.upload_pypi(RESULTS / target / report["wheel"])
        files = api.pypi_files(names)
        links = [
            {
                "name": report["wheel"],
                "url": api.verify_pypi(report, files),
                "link_type": "package",
            }
            for report in reports.values()
        ]
        current = {link["name"]: link for link in existing["assets"]["links"]} if existing else {}
        for link in links:
            if link["name"] in current and current[link["name"]]["url"] != link["url"]:
                raise ValueError("Existing Release asset points to unexpected content")
        notes_path.write_text(
            release_notes(version, commit, combinations, reports, selected), encoding="utf-8"
        )
        if existing is None:
            release = {
                "name": f"RISA Attention {version}",
                "tag_name": tag_name,
                "ref": commit,
                "tag_message": f"RISA Attention {version}",
                "description": notes_path.read_text(encoding="utf-8"),
            }
            if tag is not None:
                release.pop("ref")
                release.pop("tag_message")
            api.request("/releases", "POST", release)
        else:
            # Persist verified reports before updating links, so interrupted link
            # updates can be retried or followed by a different target's backfill.
            api.request(
                release_path, "PUT", {"description": notes_path.read_text(encoding="utf-8")}
            )
        for link in links:
            if link["name"] not in current:
                api.request(release_path + "/assets/links", "POST", link)
    (RESULTS / "SHA256SUMS").write_text(
        "".join(f"{report['sha256']}  {report['wheel']}\n" for report in reports.values()),
        encoding="utf-8",
    )
    print(notes_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "publish"])
    args = parser.parse_args()
    {"build": build, "publish": publish}[args.command]()
