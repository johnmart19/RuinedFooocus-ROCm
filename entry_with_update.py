import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.append(str(root))
os.chdir(root)

offline = os.environ.get("RF_OFFLINE") == "1" or "--offline" in sys.argv

if not offline:
    # Bootstrap before importing application helpers, which require packaging.
    print("Check pre-requirements", flush=True)
    reinstall_bootstrap = ["--force-reinstall"] if (root / "reinstall").exists() else []
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *reinstall_bootstrap, "-r", "requirements_versions.txt"],
        check=True,
    )
    # launch.py must not replace pygit2/cffi after the updater loads their DLLs.
    sys._rf_bootstrap_reinstalled = bool(reinstall_bootstrap)

    bupdated = False
    try:
        import pygit2
    
        pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)
    
        repo_path = Path(__file__).resolve().parent
        repo = pygit2.Repository(str(repo_path))
    
        branch_name = repo.head.shorthand
    
        remote_name = "origin"
        remote = repo.remotes[remote_name]
    
        remote.fetch()
    
        local_branch_ref = f"refs/heads/{branch_name}"
        local_branch = repo.lookup_reference(local_branch_ref)
    
        remote_reference = f"refs/remotes/{remote_name}/{branch_name}"
        remote_commit = repo.revparse_single(remote_reference)
    
        merge_result, _ = repo.merge_analysis(remote_commit.id)
    
        if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
            print("You have the latest version")
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            local_branch.set_target(remote_commit.id)
            repo.head.set_target(remote_commit.id)
            repo.checkout_tree(repo.get(remote_commit.id))
            repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
            print("Updating Files")
            bupdated = True
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
            print("Update failed,  Did you modify any files?")
    except Exception as e:
        print("Update failed...")
        print(str(e))
    if bupdated:
        print("Update succeeded!!")
else:
    print("Offline mode. No update.")
from launch import *
