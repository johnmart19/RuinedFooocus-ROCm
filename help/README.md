# Change python or torch version

To change the Python or PyTorch version of a portable install, copy `rf-setup.ps1` (Windows) or `rf-setup.sh` (Linux)
into the folder that contains `run.bat` and the `python_embeded` folder, then run it. You will get a simple menu.

The PyTorch choices (CUDA 12.6–13.4, AMD ROCm 10.1, CPU) install the same validated bundles as the launcher.
They check your GPU and driver first, so an unsuitable choice leaves the working installation unchanged.
A successful choice creates `freezetorch` so the launcher keeps it.

Installs made with `RuinedFooocus.bat`/`RuinedFooocus.sh` use the `venv` folder instead. There, start with
`TORCH_PLATFORM` set (for example `cu128`, `rocm10.1` or `cpu`) to switch PyTorch, or with `--rocm10`,
`--cuda-nightly`, `--cpu` or `--directml`. Add a `reinstalltorch` file to reinstall the same version.

# Manually force reinstallation

To force a reinstallation of python modules or torch, copy one of these files into the `RuinedFooocus` folder.

* reinstall - Will trigger reinstallation of all python modules
* reinstalltorch - RuinedFooocus will find a good version of pytorch for your system and reinstall it.

To freeze the torch installation and stop RuinedFooocus from "updating" it, for example if you've done it manually, copy the freezetorch file.
