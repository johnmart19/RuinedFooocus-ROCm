#!/usr/bin/env bash

# Place this script in an empty folder where you want your RuinedFooocus installation
# Make sure it is executable with: chmod 755 rf-setup.sh
# Run it: ./rf-setup.sh

python_embeded_dir="python_embeded"
python_urls=(
  "3.13.13 x86_64|https://github.com/astral-sh/python-build-standalone/releases/download/20260602/cpython-3.13.13+20260602-x86_64-unknown-linux-gnu-pgo+lto-full.tar.zst"
)
python_modules="wheel packaging pygit2 setuptools==84.2.1"
ruinedfooocus_dir="RuinedFooocus"
ruinedfooocus_repo="https://github.com/runew0lf/RuinedFooocus/"
ruinedfooocus_branches=(
  "main (recommended)"
  "development"
)
# Explicit choices use the launcher's validated, pinned wheel bundles (modules/torch_setup.py).
torch_options=(
  "Auto (Remove current Torch and let RF install its prefered version at startup)|auto"
  "CUDA 12.6 (GTX 900/1000 and newer)|cu126"
  "CUDA 12.8 (GTX 1660, RTX 20xx and newer)|cu128"
  "CUDA 13.0 (GTX 1660, RTX 20xx and newer, DGX Spark)|cu130"
  "CUDA 13.2 (latest)|cu132"
  "CUDA 13.4 (nightly)|cu134"
  "AMD ROCm 10.1 (RX 5000/6000/7000/9000, Strix Halo, Instinct MI300; GPU auto-detected)|rocm"
  "cpu|cpu"
  "Freeze current version|freeze"
  "Unfreeze (RF might update Torch automatically)|unfreeze"
)

tput civis
trap 'tput cnorm' EXIT

header() {
  echo -n "python: "
  [[ -d "$python_embeded_dir" ]] && echo -n "installed" || echo -n "missing"
  echo -n " | RuinedFooocus: "
  [[ -d "$ruinedfooocus_dir" ]] && echo -n "installed" || echo -n "missing"
  echo
  echo "Use ↑ ↓ and Enter"
  echo
}

draw_menu() {
  clear
  header
  for i in "${!options[@]}"; do
    if [[ $i == $selected ]]; then
      printf "\e[7m > %s \e[0m\n" "${options[$i]}"
    else
      printf "   %s\n" "${options[$i]}"
    fi
  done
}

menu() {
  selected=0
  while true; do
    draw_menu
    read -rsn1 key
    if [[ $key == $'\x1b' ]]; then
      read -rsn2 key

      case "$key" in
        '[A')
          ((selected--))
          ;;
        '[B')
          ((selected++))
          ;;
      esac
      ((selected<0)) && selected=$((${#options[@]}-1))
      ((selected>=${#options[@]})) && selected=0

    elif [[ $key == "" ]]; then
      break
    fi
  done
}

python_install() {
  url=$(echo "${python_urls[$1]}" | cut -d'|' -f2)
  clear
  header
  if [ -d "$python_embeded_dir" ]; then
    echo "Sorry, you already have a $python_embeded_dir folder. To change python version you need to manually remove or rename the old ${python_embeded_dir}."

    echo "Press enter to continue."
  else
    wget -O python.tar.zst "$url"
    tar xvf python.tar.zst --transform="s|^python/install|${python_embeded_dir}|" python/install
    echo "Install required modules."
    $python_embeded_dir/bin/pip install $python_modules
    echo;echo "Done... (Press enter)"
  fi
  read
}
python_loop() {
  while true; do
    options=()
    for url in "${python_urls[@]}"; do
      options+=("$(echo $url|cut -d'|' -f1)")
    done
    options+=("Back")
    menu
    if [ $selected -lt ${#python_urls[@]} ]; then
      python_install $selected
    else
      break
    fi
  done
}

git_clone() {
  clear
  header
  if [ -d "$ruinedfooocus_dir" ]; then
    echo "Sorry, you already have RuinedFooocus downloaded."
  else
    echo "Using branch ${ruinedfooocus_branches[$1]}"
    git clone -b "$(echo "${ruinedfooocus_branches[$1]}" | cut -d' ' -f1)" "$ruinedfooocus_repo"
    echo;echo "Done... (Press enter)"
  fi
  read
}
ruinedfooocus_loop() {
  while true; do
    options=()
    for branch in "${ruinedfooocus_branches[@]}"; do
      options+=("$(echo $branch|cut -d'|' -f1)")
    done
    options+=("Back")
    menu
    if [ $selected -lt ${#ruinedfooocus_branches[@]} ]; then
      git_clone $selected
    else
      break
    fi
  done
}

use_torch() {
  torch_platform=$(echo "${torch_options[$1]}" | cut -d'|' -f2)
  if [ \! -d "$python_embeded_dir" ]; then
    echo "You need to install python first."
    read
    return
  fi
  if [ \! -d "$ruinedfooocus_dir" ]; then
    echo "You need to download RuinedFooocus first."
    read
    return
  fi
  clear
  header
  case $torch_platform in
    auto)
      rm -f $ruinedfooocus_dir/freezetorch
      $python_embeded_dir/bin/pip uninstall -y torch torchvision torchaudio
      echo "Torch unfrozen"
      ;;
    freeze)
      touch $ruinedfooocus_dir/freezetorch
      echo "Torch frozen"
      ;;
    unfreeze)
      rm -f $ruinedfooocus_dir/freezetorch
      echo "Torch unfrozen"
      ;;
    *)
      # The installer validates GPU, driver and wheels before replacing the working Torch.
      echo "Install validated torch for $torch_platform"
      if (cd "$ruinedfooocus_dir" && "../$python_embeded_dir/bin/python" -m pip install --quiet -r requirements_versions.txt \
          && "../$python_embeded_dir/bin/python" -m modules.torch_setup "$torch_platform"); then
        echo "Lock torch version"
        touch $ruinedfooocus_dir/freezetorch
      else
        echo "Torch installation failed; the previous installation was kept where possible."
      fi
      ;;
  esac
  echo;echo "Done... (Press enter)"
  read
}
torch_loop() {
  while true; do
    options=()
    for torch in "${torch_options[@]}"; do
      options+=("$(echo $torch|cut -d'|' -f1)")
    done
    options+=("Back")
    menu
    if [ $selected -lt ${#torch_options[@]} ]; then
      use_torch $selected
    else
      break
    fi
  done
}
ops_loop() {
  if [ \! -d "$python_embeded_dir" ]; then
    echo "You need to install python first."
    read
    return
  fi
  if [ \! -d "$ruinedfooocus_dir" ]; then
    echo "You need to download RuinedFooocus first."
    read
    return
  fi
  clear
  header
  while true; do
    options=(
      "Fix broken python modules"
      "Trigger reinstall of all python modules next start"
      "Trigger reinstall of Torch next start (will set selected version to Auto)"
      "Back"
    )
    menu
    case $selected in
      0)
        $python_embeded_dir/bin/pip install --force-reinstall --no-cache-dir -r RuinedFooocus/requirements_versions.txt
        $python_embeded_dir/bin/pip install --force-reinstall --no-cache-dir -r RuinedFooocus/pip/modules.txt
        echo "Python modules reinstalled"
        read
        ;;
      1)
        touch RuinedFooocus/reinstall
	echo "Reinstall of python modules queued"
        read
        ;;
      2)
        echo "Removing old torch install"
        $python_embeded_dir/bin/pip uninstall -y torch torchvision torchaudio
        rm -f RuinedFooocus/freezetorch
        touch RuinedFooocus/reinstalltorch
        echo "Torch unfrozen and reinstall queued"
        read
        ;;
      *)
        break
        ;;
    esac
  done


}

main_loop() {
  while true; do
    options=(
      "Python"
      "RuinedFoocus"
      "Torch"
      "Other operations"
      "Write run.sh script"
      "Start RuinedFooocus"
      "Quit"
    )
    menu
    case $selected in
      0)
        python_loop
        ;;
      1)
        ruinedfooocus_loop
        ;;
      2)
        torch_loop
        ;;
      3)
        ops_loop
	;;
      4)
        echo "$python_embeded_dir/bin/python $ruinedfooocus_dir/entry_with_update.py" > run.sh
	chmod 755 run.sh
        echo;echo "Done... (Press enter)"
        read
        ;;
      5)
        $python_embeded_dir/bin/python $ruinedfooocus_dir/entry_with_update.py
        echo;echo "Done... (Press enter)"
        read
        ;;
      *)
        break
        ;;
    esac
  done
}


main_loop
