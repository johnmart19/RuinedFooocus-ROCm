"""Source archives exclude user data and preserve an existing artifact."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from ruinedfooocus.packaging.distribution import build_source_archive, source_checkout


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "--quiet")
        files = {
            "launch.py": 'print("source")\n',
            "ruinedfooocus/app/bootstrap.py": 'git_repos = [{"name":"ComfyUI","hash":"pinned"}]\n',
            "ruinedfooocus/resources/prompt_presets/default.txt": "Default\nKeep the scene.",
            "settings/settings.json": '{"private":true}',
            "outputs/private.png": "private",
            "models/private.safetensors": "original",
            "nested/PRIVATE.GGUF": "original",
            "AGENTS.md": "local instructions",
        }
        for name, text in files.items():
            file = self.repo / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Archive Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=" + str(self.root / "no-hooks"),
            "commit",
            "--quiet",
            "-m",
            "Source fixture",
        )

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.repo), *arguments])

    def test_archive_has_exact_source_and_resources_without_user_data_or_weights(self):
        output = self.root / "source.zip"
        (self.repo / "launch.py").write_text("uncommitted", encoding="utf-8")
        result = build_source_archive(self.repo, output)
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            self.assertEqual(archive.read("launch.py"), b'print("source")\n')
            self.assertIn("ruinedfooocus/resources/prompt_presets/default.txt", names)
            self.assertFalse(
                any(name.startswith(("settings/", "outputs/", "models/")) for name in names)
            )
            self.assertNotIn("nested/PRIVATE.GGUF", names)
            self.assertNotIn("AGENTS.md", names)
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(json.loads(archive.read("distribution.json")), result)
        self.assertEqual(result["backend_pins"]["ComfyUI"], "pinned")
        self.assertFalse(result["runtime_bundled"])
        self.assertEqual((self.repo / "models/private.safetensors").read_text(), "original")

    def test_existing_artifact_is_never_overwritten_or_deleted(self):
        output = self.root / "source.zip"
        output.write_bytes(b"keep this artifact")
        with self.assertRaises(FileExistsError):
            build_source_archive(self.repo, output)
        self.assertEqual(output.read_bytes(), b"keep this artifact")

    def test_same_revision_produces_repeatable_archive_bytes(self):
        first, second = self.root / "first.zip", self.root / "second.zip"
        build_source_archive(self.repo, first)
        build_source_archive(self.repo, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_source_detection_accepts_worktrees_and_extracted_archive_has_no_git(self):
        self.assertTrue(source_checkout(self.repo))
        folder = self.root / "worktree"
        folder.mkdir()
        (folder / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
        self.assertTrue(source_checkout(folder))
        (folder / ".git").unlink()
        self.assertFalse(source_checkout(folder))


if __name__ == "__main__":
    unittest.main()
