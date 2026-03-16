# RuinedFooocus Windows AMD gfx1100 (RX 7900 XTX) Launch Modification Instructions:
* Potentially can work with other AMD GPU-s

How to use on AMD:

* Install [Python 3.12](https://apps.microsoft.com/detail/9ncvdn91xzqp) and [Git for Windows](https://github.com/git-for-windows/git/releases/download/v2.50.1.windows.1/Git-2.50.1-64-bit.exe)
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
* `pip install -r extra-requirements.txt`

6) Install pygit2, packaging
* `pip install pygit2 packaging`

7) Launch RuinedFooocus as usual
* `python launch.py`

# After initial installation: 
Simply use `RuinedFooocus_amd_launch.bat` to launch RuinedFooocus, you can also make a shortcut for it by holding alt and moving it to your Desktop etc.
* Inside file is set --gpu-only flag by default, feel free to replace it with wanted ones

## Original RuinedFooocus Readme: https://github.com/runew0lf/RuinedFooocus/blob/main/readme.md