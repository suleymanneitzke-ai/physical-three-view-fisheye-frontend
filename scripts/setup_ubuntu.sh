#!/usr/bin/env bash
set -euo pipefail
. /etc/os-release
[[ "$ID" == ubuntu ]] || { echo "Ubuntu required: $ID" >&2; exit 2; }
for tool in cmake c++ pkg-config git python3; do command -v "$tool" >/dev/null || { echo "Missing $tool" >&2; exit 2; }; done
missing=()
for package in libopencv-dev libeigen3-dev libpcl-dev libglew-dev libgl1-mesa-dev libgtk-3-dev; do
  dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed" || missing+=("$package")
done
if (( ${#missing[@]} )); then
  echo "Missing packages: ${missing[*]}" >&2
  [[ "${INSTALL_SYSTEM_PACKAGES:-0}" == 1 ]] || { echo "Set INSTALL_SYSTEM_PACKAGES=1 for interactive sudo installation" >&2; exit 2; }
  read -r -p "Install listed packages using sudo? [y/N] " answer
  [[ "$answer" == y || "$answer" == Y ]] || exit 2
  sudo apt update
  sudo apt install --no-install-recommends "${missing[@]}"
fi
if [[ -z "${Pangolin_DIR:-}" ]] && ! find /usr /opt "$HOME/.local" -name PangolinConfig.cmake -print -quit 2>/dev/null | grep -q .; then
  echo "Pangolin v0.9.5 CMake package missing; install upstream and set Pangolin_DIR" >&2
  exit 2
fi
printf "SETUP_CHECK_PASS Ubuntu=%s OpenCV=%s\n" "$VERSION_ID" "$(pkg-config --modversion opencv4)"
