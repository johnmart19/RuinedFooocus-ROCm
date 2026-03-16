# RuinedFooocus Windows AMD AI Bundle Launch Modification Instructions:
* Tested with Sapphire Radeon RX 7900 XTX Nitro+ Vapor-X

How to use on AMD:
* Install latest AMD Driver with AI Bundle (Pytorch bundle is required):
Example for RX 7900 XTX: https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-7000-series/amd-radeon-rx-7900-xtx.html
Tested with driver version: Adrenalin 26.6.4 (WHQL Recommended) | torch 2.9.1+rocmsdk20260116
* If errors occur - Install [Microsoft C++ Build Tools for Nexa AI Dependency compilation](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Installation:
1) Clone Repo
* `git clone https://github.com/johnmart19/RuinedFooocus RuinedFooocus && cd RuinedFooocus`
2) Update pip (Optional)
* `python.exe -m pip install --upgrade pip`
3) Launch RuinedFooocus as usual
* `python launch.py`

Recommended:
Install Vulkan version of xllamacpp for fast LLM processing for Chat bots:
* `python -m pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan`

# After initial installation:
Simply use `RuinedFooocus_amd_launch.bat` to launch RuinedFooocus, you can also make a shortcut for it by holding alt and moving it to your Desktop etc.
* Inside file is set --gpu-only flag by default, feel free to replace it with wanted ones

## Original RuinedFooocus Readme: https://github.com/runew0lf/RuinedFooocus/blob/main/readme.md