# Launchers

On Windows, run `RuinedFooocus.bat` using Python from your active environment or PATH.
On Linux/WSL, run `./RuinedFooocus.sh`. It creates `venv` on first launch using
`python3`, then activates it. Existing environments are reused. To choose another
Python for initial setup, use `PYTHON=python3.12 ./RuinedFooocus.sh`.
If Ubuntu cannot create the environment, install `python3-venv` (or the package
matching your chosen Python version), then rerun the launcher. Application
dependencies are installed by the normal startup process.
Both scripts forward arguments to `entry_with_update.py` and work with NVIDIA or AMD.
