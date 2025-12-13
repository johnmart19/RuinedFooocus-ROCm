# RuinedFooocus Windows AMD gfx1100 (RX 7900 XTX) Launch Modification Instructions:
* Potentially can work with other gfx110x series
* For gfx1151 and gfx1201 download cp312 files from [rocm-TheRock repository](https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch)

How to use on AMD:

* Install [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) as usual and [Git for Windows](https://github.com/git-for-windows/git/releases/download/v2.50.1.windows.1/Git-2.50.1-64-bit.exe)
* Install [Microsoft C++ Build Tools for Nexa AI Dependency compilation](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Installation:
1) Clone Repo
* `git clone https://github.com/johnmart19/RuinedFooocus RuinedFooocus && cd RuinedFooocus`
2) Make Venv
* `python -m venv venv`
3) Activate it
* `.\venv\Scripts\activate`
4) Update pip (Optional)
* `python.exe -m pip install --upgrade pip`
5) Install appropriate AMD Torch versions (Mandatory):

Option 1 (Latest by AMD):
* `python -m pip install  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/  "rocm[libraries,devel]"`
* `python -m pip install  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/  --pre torch torchaudio torchvision`

Option 2 (rocm-TheRock version, Not updated):
* ` pip install -r extra-requirements.txt`
6) Launch RuinedFooocus as usual
* `python launch.py`

# After initial installation: 
Simply use `RuinedFooocus_amd_launch.bat` to launch RuinedFooocus, you can also make a shortcut for it by holding alt and moving it to your Desktop etc.
* Inside file is set --gpu-only flag by default, feel free to replace it with wanted ones

## Original RuinedFooocus Readme: https://github.com/runew0lf/RuinedFooocus/blob/main/readme.md