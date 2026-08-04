# RuinedFooocus Windows AMD AI Bundle Launch Modification Instructions:
* Tested with Sapphire Radeon RX 7900 XTX Nitro+ Vapor-X

How to use on AMD:
* Install latest AMD Driver with HIP SDK (optional):
Example for RX 7900 XTX: https://www.amd.com/en/developer/resources/rocm-hub/eula/licenses.html?filename=AMD-Software-PRO-Edition-26.Q3-Win11-For-HIP.exe.exe
Tested with driver version: Adrenalin 26.10.21.06 (PRO Edition) | Torch version: 2.10.0+rocm7.2.4.lw.git3d3aa833

Installation:
1) Clone Repo
* `git clone https://github.com/johnmart19/RuinedFooocus RuinedFooocus && cd RuinedFooocus`
2) Make Venv
* `python -m venv venv`
3) Activate it
* `source venv/bin/activate`
4) Update pip (Optional)
* `python.exe -m pip install --upgrade pip`
5) Install appropriate AMD Torch versions (Mandatory):
* `pip install -r extra-requirements-linux.txt`

6) Install pygit2, packaging (if missing)
* `pip install pygit2 packaging`

7) Launch RuinedFooocus as usual
* `python launch.py`

# After initial installation:
Simply use `RuinedFooocus_amd_launch.sh` to launch RuinedFooocus, you can also make a shortcut for it by holding alt and moving it to your Desktop etc.
* Inside file is set --gpu-only flag by default, feel free to replace it with wanted ones

## Original RuinedFooocus Readme: https://github.com/runew0lf/RuinedFooocus/blob/main/readme.md
