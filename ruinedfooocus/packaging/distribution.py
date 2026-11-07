"""Source archive preparation; platform runtime bundles are separate releases."""

import ast
import json
from pathlib import Path
import subprocess
import zipfile

PORTABLE_TARGETS = ("windows-cuda", "windows-rocm", "linux-cuda", "linux-rocm")
USER_DIRECTORIES = (
    "settings",
    "outputs",
    "models",
    "cache",
    "venv",
    "repositories",
    "tmp",
    "chatbots",
    "llamas",
    "assistants",
    "prompt_presets",
    "history-backups",
    ".codex",
    ".claude",
    ".agents",
)
WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".gguf", ".pth", ".pt", ".onnx", ".bin")


def source_checkout(root):
    """.git is a file in a worktree, a directory in a conventional clone."""
    return (Path(root) / ".git").exists()


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def build_source_archive(root, output, ref="HEAD"):
    root, output = Path(root).resolve(), Path(output).resolve()
    commit = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    bootstrap = git(root, "show", f"{commit}:ruinedfooocus/app/bootstrap.py").decode("utf-8")
    tree = ast.parse(bootstrap)
    repositories = ast.literal_eval(
        next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(name, ast.Name) and name.id == "git_repos" for name in node.targets)
        )
    )
    manifest = {
        "kind": "source-archive",
        "source_commit": commit,
        "backend_pins": {item["name"]: item["hash"] for item in repositories},
        "portable_targets": list(PORTABLE_TARGETS),
        "runtime_bundled": False,
        "models_bundled": False,
        "data_environment": "RF_DATA_DIR",
    }
    exclusions = [f":(exclude){name}" for name in USER_DIRECTORIES]
    exclusions += [f":(exclude,icase)*{suffix}" for suffix in WEIGHT_SUFFIXES]
    exclusions += [":(exclude)AGENTS.md", ":(exclude)CLAUDE.md", ":(exclude)distribution.json"]
    exclusions += [f":(exclude){name}" for name in ("reinstall", "reinstalltorch", "freezetorch")]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is outside cleanup: an existing artifact is never removed.
    destination = output.open("x+b")
    try:
        with destination:
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "-C",
                    str(root),
                    "archive",
                    "--format=zip",
                    commit,
                    "--",
                    ".",
                    *exclusions,
                ],
                stdout=destination,
                check=True,
            )
            with zipfile.ZipFile(destination, "a", compression=zipfile.ZIP_DEFLATED) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("Archive contains duplicate paths")
                if any(name.lower().endswith(WEIGHT_SUFFIXES) for name in names):
                    raise ValueError("Source archives must not include model weights")
                entry = zipfile.ZipInfo(
                    "distribution.json", date_time=archive.infolist()[0].date_time
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return manifest
