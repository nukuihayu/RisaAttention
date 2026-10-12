"""Publication behavior, including interrupted uploads and partial backfills."""

import importlib.util
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("wheel_release", ROOT / "ci/wheels.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
SHA = "a" * 40


@pytest.fixture
def environment(tmp_path, monkeypatch):
    (tmp_path / "VERSION").write_text("0.2.0.dev1\n")
    shutil.copyfile(ROOT / ".gitlab-ci.yml", tmp_path / ".gitlab-ci.yml")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "RESULTS", tmp_path / "results")
    monkeypatch.setenv("CI_COMMIT_SHA", SHA)
    monkeypatch.setenv("CI_COMMIT_BRANCH", "codex/test")
    monkeypatch.setenv("CI_DEFAULT_BRANCH", "main")
    monkeypatch.delenv("CI_COMMIT_TAG", raising=False)
    monkeypatch.setenv("PUBLISH", "false")
    monkeypatch.setenv("BUILD_TARGET", "all")
    monkeypatch.setenv("PUBLISH_ONLY", "false")
    monkeypatch.delenv("PUBLISH_ARTIFACT_JOBS", raising=False)
    monkeypatch.delenv("RELEASE_COMMIT_SHA", raising=False)
    return tmp_path


def wheel_result(python="3.12"):
    config = release.PACKAGE
    target = config["distribution"] + "-cu130-py" + python.replace(".", "")
    directory = release.RESULTS / target
    directory.mkdir(parents=True, exist_ok=True)
    abi = "cp" + python.replace(".", "")
    version = release.base_version() + "+cu130"
    normalized = config["distribution"].replace("-", "_")
    name = f"{normalized}-{version}-{abi}-{abi}-linux_x86_64.whl"
    wheel = directory / name
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{normalized}-{version}.dist-info/METADATA",
            f"Name: {config['distribution']}\nVersion: {version}\n",
        )
        archive.writestr(
            f"{config['module']}/{config['extension']}.{abi}-x86_64-linux-gnu.so", b"test extension"
        )
        archive.writestr(config["module"] + "/_version.py", f'__version__ = "{version}"\n')
    report = {
        "version": release.base_version(),
        "commit": SHA,
        "target": target,
        "package": config["distribution"],
        "cuda": "cu130",
        "python": python,
        "architectures": ["80-real", "120-real"],
        "image": release.matrix()[target]["CUDA_IMAGE"],
        "wheel": name,
        "sha256": release.sha256(wheel),
    }
    (directory / "result.json").write_text(json.dumps(report))
    return report


class Registry:
    base = "https://example.invalid/api/v4/projects/1"

    def __init__(self):
        self.files = {}
        self.release = None
        self.tag = None
        self.fail_marker = False
        self.writes = []
        self.uploads = []

    def find_tag(self, name):
        return self.tag

    def pypi_files(self, names):
        return {
            name: {
                "sha256": release.hash_stream(io.BytesIO(data)),
                "url": self.base
                + "/packages/pypi/files/"
                + release.hash_stream(io.BytesIO(data))
                + "/"
                + release.quote(name, safe=""),
            }
            for name, data in self.files.items()
            if name in names
        }

    def upload_pypi(self, source):
        assert source.name not in self.files
        self.files[source.name] = source.read_bytes()
        self.uploads.append(source.name)

    def verify_pypi(self, report, files):
        if self.fail_marker:
            raise RuntimeError("simulated interrupted upload")
        assert files[report["wheel"]]["sha256"] == report["sha256"]
        return files[report["wheel"]]["url"]

    def request(self, path, method="GET", data=None):
        if method == "GET":
            return json.loads(json.dumps(self.release))
        self.writes.append((path, method, data))
        if path == "/releases":
            if self.tag is None:
                self.tag = {"commit": {"id": data["ref"]}}
            self.release = {
                "assets": {"links": []},
                "description": data["description"],
                "commit": self.tag["commit"],
            }
        elif path.endswith("/assets/links"):
            self.release["assets"]["links"].append(
                {**data, "id": len(self.release["assets"]["links"]) + 1}
            )
        else:
            self.release["description"] = data["description"]
        return {}


def test_explicit_architectures_require_sm80_native_targets():
    assert release.architectures("80-real;120-real") == ["80-real", "120-real"]
    for value in [
        "",
        "native",
        "all",
        "75-real;120-real",
        "90a-real",
        "100f-real",
        "120",
        "80-real;80-real",
    ]:
        with pytest.raises(ValueError):
            release.architectures(value)


def test_build_only_partial_success_has_no_network(environment, monkeypatch):
    wheel_result()
    monkeypatch.setattr(
        release, "GitLab", lambda: pytest.fail("Build-only must not contact GitLab")
    )
    release.publish()
    notes = (release.RESULTS / "release-notes.md").read_text(encoding="utf-8")
    assert "| risa-attention-cu130-py312 | Available" in notes
    assert "| risa-attention-cu130-py313 | Failed or missing" in notes


def test_build_passes_explicit_architectures_without_discovery(environment, monkeypatch):
    monkeypatch.setenv("CUDA_TAG", "cu130")
    monkeypatch.setenv("PYTHON_VERSION", "3.13")
    monkeypatch.setenv("CUDACXX", "nvcc")
    monkeypatch.setenv("RISA_CUDA_ARCHITECTURES", "80-real;120-real")
    monkeypatch.setattr(release.sys, "version_info", (3, 13))

    def compiler(command, **kwargs):
        assert command == ["nvcc", "--version"]  # No GPU or nvcc SM discovery.
        return "Cuda compilation tools, release 13.0, V13.0.88"

    def build(command, *, env, check):
        assert env["RISA_CUDA_ARCHITECTURES"] == "80-real;120-real"
        assert env["CMAKE_BUILD_PARALLEL_LEVEL"] == "1"
        assert command[2] == "--wheel"
        wheel_result("3.13")

    monkeypatch.setattr(release.subprocess, "check_output", compiler)
    monkeypatch.setattr(release.subprocess, "run", build)
    release.build()
    report = json.loads(
        (release.RESULTS / "risa-attention-cu130-py313/result.json").read_text(encoding="utf-8")
    )
    assert report["architectures"] == ["80-real", "120-real"]


def test_all_failed_never_creates_release(environment, monkeypatch):
    monkeypatch.setattr(release, "GitLab", lambda: pytest.fail("No artifacts must not publish"))
    with pytest.raises(ValueError, match="No successful"):
        release.publish()


def test_backfill_is_idempotent_and_preserves_first_wheel(environment, monkeypatch):
    registry = Registry()
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    monkeypatch.setenv("BUILD_TARGET", "risa-attention-cu130-py312")
    first = wheel_result()
    release.publish()
    assert registry.tag["commit"]["id"] == SHA
    release.publish()
    assert len(registry.release["assets"]["links"]) == 1
    assert registry.uploads == [first["wheel"]]
    monkeypatch.setenv("BUILD_TARGET", "risa-attention-cu130-py313")
    second = wheel_result("3.13")
    release.publish()
    assert {link["name"] for link in registry.release["assets"]["links"]} == {
        first["wheel"],
        second["wheel"],
    }
    assert "Failed or missing" not in (release.RESULTS / "release-notes.md").read_text(
        encoding="utf-8"
    )
    assert len((release.RESULTS / "SHA256SUMS").read_text(encoding="utf-8").splitlines()) == 2
    create = next(
        data for path, method, data in registry.writes if path == "/releases" and method == "POST"
    )
    assert "assets" not in create
    assert all("direct_asset_path" not in link for link in registry.release["assets"]["links"])
    assert len(registry.uploads) == 2
    persisted = release.previous_reports(
        registry.release, release.matrix(), release.base_version(), SHA, {}
    )
    assert set(persisted) == {first["target"], second["target"]}


def test_interrupted_upload_can_resume_without_early_tag(environment, monkeypatch):
    registry = Registry()
    registry.fail_marker = True
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    wheel_result()
    with pytest.raises(RuntimeError, match="interrupted"):
        release.publish()
    assert registry.tag is None
    registry.fail_marker = False
    release.publish()
    assert registry.tag["commit"]["id"] == SHA


def test_release_creation_resumes_after_tag_was_already_created(environment, monkeypatch):
    registry = Registry()
    registry.tag = {"commit": {"id": SHA}}
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    wheel_result()
    release.publish()
    create = next(
        data for path, method, data in registry.writes if path == "/releases" and method == "POST"
    )
    assert "ref" not in create
    assert "tag_message" not in create
    assert len(registry.release["assets"]["links"]) == 1


def test_same_version_different_commit_is_rejected_before_upload(environment, monkeypatch):
    registry = Registry()
    registry.tag = {"commit": {"id": "b" * 40}}
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    wheel_result()
    with pytest.raises(ValueError, match="bump VERSION"):
        release.publish()
    assert not registry.files


def test_tampered_wheel_and_stable_feature_branch_cannot_publish(environment, monkeypatch):
    report = wheel_result()
    wheel = release.RESULTS / report["target"] / report["wheel"]
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("unexpected.py", "changed")
    with pytest.raises(ValueError, match="checksum"):
        release.publish()
    (environment / "VERSION").write_text("0.2.0")
    monkeypatch.setenv("PUBLISH", "true")
    with pytest.raises(ValueError, match="default branch"):
        release.publish()


def test_stable_release_can_be_backfilled_from_its_tag_after_main_moves(environment, monkeypatch):
    registry = Registry()
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    monkeypatch.setenv("CI_COMMIT_BRANCH", "main")
    monkeypatch.setenv("BUILD_TARGET", "risa-attention-cu130-py312")
    (environment / "VERSION").write_text("0.2.0")
    wheel_result()
    release.publish()
    monkeypatch.delenv("CI_COMMIT_BRANCH")
    monkeypatch.setenv("CI_COMMIT_TAG", "v0.2.0")
    monkeypatch.setenv("BUILD_TARGET", "risa-attention-cu130-py313")
    wheel_result("3.13")
    release.publish()
    assert len(registry.release["assets"]["links"]) == 2
    registry.release = None
    registry.writes.clear()
    with pytest.raises(ValueError, match="only backfill"):
        release.publish()
    assert not registry.writes


def test_standalone_sdist_keeps_version_without_parent_checkout(tmp_path):
    import subprocess
    import sys

    project = tmp_path / "repo" / "risa_attention"
    project.parent.mkdir(parents=True)
    shutil.copytree(
        ROOT,
        project,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "build", "dist", ".pytest_cache", ".ruff_cache"
        ),
    )
    (project / "VERSION").write_text("0.9.1rc2")
    output = tmp_path / "sdist"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(output),
            str(project),
        ],
        check=True,
    )
    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        archive.extractall(tmp_path / "standalone", filter="data")
    standalone = next((tmp_path / "standalone").iterdir())
    (tmp_path / "VERSION").write_text("9.9.9")  # unrelated surrounding project
    assert (standalone / "VERSION").read_text(encoding="utf-8").strip() == "0.9.1rc2"
    assert "Version: 0.9.1rc2" in (standalone / "PKG-INFO").read_text(encoding="utf-8")
    assert not (standalone / "version_provider.py").exists()


def test_pypi_index_and_download_verification(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    uploaded = {}
    corrupt = False
    name = "risa_attention-0.1.0+cu130-cp312-cp312-linux_x86_64.whl"
    source = tmp_path / name
    source.write_bytes(bytes(range(256)) * 5000)
    digest = release.sha256(source)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            expected = release.base64.b64encode(b"gitlab-ci-token:local-test-token").decode()
            assert self.headers["Authorization"] == "Basic " + expected
            self.send_response(200 if uploaded else 404)
            self.end_headers()
            if uploaded:
                if "/simple/" in self.path:
                    self.wfile.write(
                        (
                            f'<a href="{api.base}/packages/pypi/files/{digest}/'
                            f'{release.quote(name)}#sha256={digest}">{name}</a>'
                            '<a href="old_version.whl" data-yanked="">old version</a>'
                        ).encode()
                    )
                else:
                    self.wfile.write(b"corrupt" if corrupt else uploaded[name])

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("CI_API_V4_URL", f"http://127.0.0.1:{server.server_port}/api/v4")
    monkeypatch.setenv("CI_PROJECT_ID", "1")
    monkeypatch.setenv("CI_JOB_TOKEN", "local-test-token")
    try:
        api = release.GitLab()
        assert api.pypi_files({name}) == {}
        uploaded[name] = source.read_bytes()
        files = api.pypi_files({name})
        assert set(files) == {name}
        report = {"wheel": name, "sha256": digest}
        assert api.verify_pypi(report, files) == files[name]["url"]
        with pytest.raises(ValueError, match="Refusing to replace"):
            api.verify_pypi({"wheel": name, "sha256": "0" * 64}, files)
        corrupt = True
        with pytest.raises(ValueError, match="checksum mismatch"):
            api.verify_pypi(report, files)
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_find_tag_uses_authenticated_git_remote(environment, monkeypatch):
    monkeypatch.setenv("CI_API_V4_URL", "https://example.invalid/api/v4")
    monkeypatch.setenv("CI_PROJECT_ID", "1")
    monkeypatch.setenv("CI_JOB_TOKEN", "test-token")
    tag_object = "b" * 40

    def ls_remote(command, *, text):
        assert command == [
            "git",
            "ls-remote",
            "--tags",
            "origin",
            "refs/tags/v0.2.0.dev1",
            "refs/tags/v0.2.0.dev1^{}",
        ]
        assert text is True
        return f"{tag_object}\trefs/tags/v0.2.0.dev1\n{SHA}\trefs/tags/v0.2.0.dev1^{{}}\n"

    monkeypatch.setattr(release.subprocess, "check_output", ls_remote)
    assert release.GitLab().find_tag("v0.2.0.dev1")["commit"]["id"] == SHA


def test_publish_job_installs_git_before_tag_lookup():
    config = release.yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8"))
    commands = config["publish:release"]["before_script"]
    assert "apt-get install -y --no-install-recommends git" in commands
    assert config["publish:release"]["script"][-1] == "python ci/wheels.py publish"


def test_publish_only_downloads_old_artifacts_and_uses_their_commit(environment, monkeypatch):
    old_commit = "c" * 40
    monkeypatch.setenv("PUBLISH", "true")
    monkeypatch.setenv("PUBLISH_ONLY", "true")
    monkeypatch.setenv("PUBLISH_ARTIFACT_JOBS", "1742")
    monkeypatch.setenv("RELEASE_COMMIT_SHA", old_commit)
    monkeypatch.setenv("CI_API_V4_URL", "https://example.invalid/api/v4")
    monkeypatch.setenv("CI_PROJECT_ID", "1")
    monkeypatch.setenv("CI_JOB_TOKEN", "test-token")
    report = wheel_result()
    report["commit"] = old_commit
    result_path = release.RESULTS / report["target"] / "result.json"
    result_path.write_text(json.dumps(report))
    artifact_data = io.BytesIO()
    with zipfile.ZipFile(artifact_data, "w") as artifact:
        for path in (release.RESULTS / report["target"]).iterdir():
            artifact.write(path, ".gitlab-results/" + report["target"] + "/" + path.name)
    shutil.rmtree(release.RESULTS)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    monkeypatch.setattr(
        release, "urlopen", lambda request, timeout: Response(artifact_data.getvalue())
    )
    registry = Registry()
    api = release.GitLab()
    api.request = registry.request
    api.pypi_files = registry.pypi_files
    api.upload_pypi = registry.upload_pypi
    api.verify_pypi = registry.verify_pypi
    api.find_tag = registry.find_tag
    monkeypatch.setattr(release, "GitLab", lambda: api)
    release.publish()
    assert registry.tag["commit"]["id"] == old_commit
    assert (release.RESULTS / report["target"] / report["wheel"]).is_file()


@pytest.mark.parametrize("missing_manifest", [True, False])
def test_existing_release_conflicts_are_not_repaired(environment, monkeypatch, missing_manifest):
    wheel_result()
    registry = Registry()
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    release.publish()
    if missing_manifest:
        registry.release["description"] = "Release notes without a build manifest"
        message = "Malformed release manifest"
    else:
        registry.release["assets"]["links"][0]["url"] = "https://unexpected.invalid/wheel"
        message = "Existing Release asset points to unexpected content"
    before = json.dumps(registry.release, sort_keys=True)
    registry.writes.clear()
    registry.uploads.clear()
    with pytest.raises(ValueError, match=message):
        release.publish()
    assert not registry.writes
    assert not registry.uploads
    assert json.dumps(registry.release, sort_keys=True) == before


def test_conflicting_pypi_file_fails_before_any_new_upload(environment, monkeypatch):
    wheel_result()
    second = wheel_result("3.13")
    registry = Registry()
    registry.files[second["wheel"]] = b"different build"
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    with pytest.raises(ValueError, match="Refusing to replace"):
        release.publish()
    assert registry.uploads == []
    assert registry.tag is None


def test_changed_report_and_missing_previous_wheel_block_backfill(environment, monkeypatch):
    first = wheel_result()
    registry = Registry()
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    release.publish()
    path = release.RESULTS / first["target"] / "result.json"
    changed = {**first, "architectures": ["120-real"]}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="published build report"):
        release.publish()
    wheel_result("3.13")
    monkeypatch.setenv("BUILD_TARGET", "cu130-py313")
    registry.files.clear()
    with pytest.raises(ValueError, match="Previously released wheel missing"):
        release.publish()
    assert len(registry.uploads) == 1


def test_interrupted_release_links_preserve_manifest_for_next_backfill(environment, monkeypatch):
    first = wheel_result()
    registry = Registry()
    request = registry.request

    def interrupted(path, method="GET", data=None):
        if path.endswith("/assets/links") and method == "POST":
            raise RuntimeError("link update interrupted")
        return request(path, method, data)

    registry.request = interrupted
    monkeypatch.setattr(release, "GitLab", lambda: registry)
    monkeypatch.setenv("PUBLISH", "true")
    with pytest.raises(RuntimeError, match="link update interrupted"):
        release.publish()
    assert release.MANIFEST_START in registry.release["description"]
    registry.request = request
    second = wheel_result("3.13")
    monkeypatch.setenv("BUILD_TARGET", "cu130-py313")
    release.publish()
    assert {link["name"] for link in registry.release["assets"]["links"]} == {
        first["wheel"],
        second["wheel"],
    }
    assert len(registry.uploads) == 2


@pytest.mark.parametrize(
    "href",
    [
        "https://external.invalid/package.whl#sha256=" + "a" * 64,
        "/api/v4/projects/2/packages/pypi/files/" + "a" * 64 + "/package.whl#sha256=" + "a" * 64,
        "/api/v4/projects/1/packages/pypi/files/" + "a" * 64 + "/package.whl",
    ],
)
def test_pypi_index_rejects_external_unhashed_and_other_project_links(href):
    parser = release.WheelIndex("https://example.invalid/api/v4/projects/1", {"package.whl"})
    with pytest.raises(ValueError, match="Unexpected PyPI"):
        parser.feed(f'<a href="{href}">wheel</a>')
