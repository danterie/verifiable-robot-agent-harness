#!/usr/bin/env bash
# Ubuntu 24.04 ROS 2 Jazzy + MoveIt 2. Needs sudo (password prompt).
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

export DEBIAN_FRONTEND=noninteractive
. /etc/os-release
if [[ "${VERSION_CODENAME}" != "noble" ]]; then
  echo "This script targets Ubuntu 24.04 (noble). Found: ${VERSION_CODENAME}" >&2
  exit 1
fi

apt-get update
apt-get install -y software-properties-common curl gnupg lsb-release locales
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

add-apt-repository -y universe
ROS_APT_MIRROR="${ROS_APT_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu}"
curl -4 -fsSL --connect-timeout 10 --max-time 30 \
  https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
chmod 0644 /usr/share/keyrings/ros-archive-keyring.gpg
printf 'deb [arch=%s signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] %s %s main\n' \
  "$(dpkg --print-architecture)" "${ROS_APT_MIRROR}" "${VERSION_CODENAME}" \
  > /etc/apt/sources.list.d/ros2.list
apt-get update
apt-get install -y \
  ros-jazzy-desktop \
  ros-dev-tools \
  ros-jazzy-moveit \
  ros-jazzy-rmw-cyclonedds-cpp \
  python3-pip python3-venv python3-dev \
  build-essential cmake ninja-build pkg-config

echo
echo "ROS 2 Jazzy installed. In each new shell run:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
