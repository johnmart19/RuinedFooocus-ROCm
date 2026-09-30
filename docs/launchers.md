# Launchers

On Windows, run `RuinedFooocus.bat`. It uses an explicit `PYTHON` executable,
an activated virtual environment, or the checkout's `venv`, in that order.
For a new checkout it creates `venv` using a working `python` on PATH, falling
back to `py -3`. Python 3.10 or newer must be installed first; a broken Microsoft
Store alias does not count as a working interpreter. After installing Python,
reopen your terminal so the updated PATH is available. To select an existing
environment explicitly, use `set "PYTHON=C:\path\to\python.exe"` before launching.
Both launchers change to their own folder before starting the application.
On Linux/WSL, run `./RuinedFooocus.sh`. It creates `venv` on first launch using
`python3`, then activates it. Existing environments are reused. To choose another
Python for initial setup, use `PYTHON=python3.12 ./RuinedFooocus.sh`.
If Ubuntu cannot create the environment, install `python3-venv` (or the package
matching your chosen Python version), then rerun the launcher. Application
dependencies are installed by the normal startup process.
Both scripts forward arguments to `entry_with_update.py` and work with NVIDIA or AMD.

## LAN access and guests

By default the UI listens on `127.0.0.1`, accessible only on the host. For your
own devices on a trusted network, set a UI password and add `--listen`:

```bat
set "RF_UI_AUTH=owner/choose-a-long-password"
RuinedFooocus.bat --listen 0.0.0.0 --port 7863
```

On Linux/WSL, export `RF_UI_AUTH`, then run
`./RuinedFooocus.sh --listen 0.0.0.0 --port 7863`. Open
`http://<host-LAN-IP>:7863` on the other device. Allow the port only on your
trusted/private firewall profile; WSL may also need host networking/forwarding.
The launcher does not change firewall rules or router port forwarding.
HTTP is not encrypted: use an HTTPS reverse proxy or VPN outside a trusted LAN.
Set authentication even when a reverse proxy connects to a loopback listener.

For another person, add **`--guest`** and use a different guest password.
This starts a generation-only interface instead of the owner UI. It exposes
installed image/video models, performance presets, resolution and blank prompts.
It does not register the owner's gallery, search, last-image, settings, chat,
OpenAI API or MCP endpoints. `--guest --api` and `--guest --mcp` are rejected.
Run one GPU instance at a time; restart without `--guest` for the full owner UI.

Guest results are saved under your configured outputs folder in
`guests/<random-run-id>/`. They appear only in the generating browser and are
served through URLs bound to that browser's authenticated login. A second login
cannot retrieve those URLs, even using the same username/password. Result links
last at most 24 hours, and expire on server restart. Logging out and back in creates a new
login. The host can still inspect the saved files; this is not privacy from the host.

There is **no NSFW classifier**. Guest mode hides *all* existing outputs, rather
than trying to guess which are adult content. New results are shown to their
requester regardless of subject. Do not give guests the owner login
or put the full owner UI on a public endpoint: its users share settings and history.

Public Gradio tunnels (`--share`) and built-in image publishing buttons are
disabled. Responses use `Cache-Control: private, no-store`, and Gradio telemetry
is disabled. No automatic image publication is performed. Guest results are
served directly without additional Gradio image copies; the owner UI still uses
Gradio temporary files for previews/uploads and saves outputs and metadata locally.
This does not erase existing caches, browser downloads, screenshots, backups,
proxy logs, OS paging or saved outputs. It does not promise zero disk traces.
See [Gradio file access](https://www.gradio.app/guides/file-access) for the
temporary-file behavior. `--clean-cache` clears the legacy Gradio temporary cache
on startup; it does not delete outputs or model caches.

## Occupied ports

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
