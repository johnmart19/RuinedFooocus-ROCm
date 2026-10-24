# Change python or torch version

If you want to change python version or change configuration, copy the `rf-setup.ps1` (Windows) or `rf-setup.sh` (Linux)
script to the same folder as `run.bat` and the `python_embedded` folder and then run it. You will get a simple menu with easy to follow options.

# Manually force reinstallation

To force a reinstallation of python modules or torch; copy one of the reinstall files to the `RuinedFoocus` folder.

* reinstall - Will trigger reinstallation of all python modules
* reinstalltorch - RouinedFooocus will try to find a good version of pytorch for your system and reinstall it.

To freeze the torchinstallation and stop RuinedFooocus from "updating" it, for example if you've done it manually, copy the freezetorch file.
