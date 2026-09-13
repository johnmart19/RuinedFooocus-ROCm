# Launchers

On Windows, run `RuinedFooocus.bat` using Python from your active environment or PATH.
On Linux/WSL, run `./RuinedFooocus.sh`. It creates `venv` on first launch using
`python3`, then activates it. Existing environments are reused. To choose another
Python for initial setup, use `PYTHON=python3.12 ./RuinedFooocus.sh`.
If Ubuntu cannot create the environment, install `python3-venv` (or the package
matching your chosen Python version), then rerun the launcher. Application
dependencies are installed by the normal startup process.
Both scripts forward arguments to `entry_with_update.py` and work with NVIDIA or AMD.

If an explicit `--port` is occupied, startup offers another port or cancellation.
The UI and API share a port; the API remains enabled if requested. The choice applies only to
this launch; update API clients to use the new address. Without `--port`, an
available nearby port is selected automatically. With no terminal input, an
explicit port conflict exits with a short message; set an available `--port`
when launching unattended.

If the port belongs to a Python instance launched from this RuinedFooocus folder,
startup first asks whether to close it and reuse the port (default: No). Yes
terminates that instance, including active work and connected sessions. Other
applications, unrecognized launch commands, and instances whose identity cannot
be checked are never offered for termination. No keeps the existing instance
running and continues with the port-selection options.
