import os
import subprocess
import sys
from pathlib import Path
from ruinedfooocus.packaging.distribution import source_checkout
from ruinedfooocus.core.paths import startup_marker

root = Path(__file__).resolve().parent
sys.path.append(str(root))
os.chdir(root)

offline = os.environ.get("RF_OFFLINE") == "1" or "--offline" in sys.argv

if not offline:
    from ruinedfooocus.core.dependency_cache import needed, record

    # Bootstrap before importing application helpers, which require packaging.
    reinstall_bootstrap = ["--force-reinstall"] if startup_marker("reinstall").exists() else []
    if needed(root, "bootstrap", force=bool(reinstall_bootstrap)):
        print("Check pre-requirements", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                *reinstall_bootstrap,
                "-r",
                "requirements_versions.txt",
            ],
            check=True,
        )
        record(root, "bootstrap")
    sys._rf_bootstrap_checked = True
    # ruinedfooocus/app/bootstrap.py must not replace pygit2/cffi after the updater loads their DLLs.

if not offline and source_checkout(root):
    bupdated = False
    try:
        import pygit2

        pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

        repo_path = Path(__file__).resolve().parent
        repo = pygit2.Repository(str(repo_path))
        if repo.status():
            raise RuntimeError(
                "Local source changes found. Keeping them; automatic update skipped."
            )
        if repo.head_is_detached:
            raise RuntimeError("Detached source revision; automatic update skipped.")
        expected_head = (repo.head.name, repo.head.target)
        branch_name = repo.head.shorthand

        remote_name = "origin"
        remote = repo.remotes[remote_name]

        remote.fetch()

        remote_reference = f"refs/remotes/{remote_name}/{branch_name}"
        remote_commit = repo.revparse_single(remote_reference)

        merge_result, _ = repo.merge_analysis(remote_commit.id)

        if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
            print("You have the latest version")
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            from ruinedfooocus.packaging.source_updates import fast_forward

            fast_forward(repo, remote_commit, expected_head)
            print("Updating Files")
            bupdated = True
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
            print("Update failed,  Did you modify any files?")
    except Exception as e:
        print("Update failed...")
        print(str(e))
    if bupdated:
        print("Update succeeded!!")
        # Check the new dependency files in a fresh process, before updater
        # imports can hold native package DLLs open on Windows.
        os.execv(sys.executable, [sys.executable, *sys.argv])
else:
    print(
        "Offline mode. No update."
        if offline
        else "Archive installation: Git updates are disabled. Replace application files to upgrade."
    )
from ruinedfooocus.app.bootstrap import *
