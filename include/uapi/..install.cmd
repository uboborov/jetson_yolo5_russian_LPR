cmd_/home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/uapi/.install := /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/uapi ./include/uapi ; /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/uapi ./include ; /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/uapi ./include/generated/uapi ; for F in ; do echo "$(pound)include <asm-generic/$$F>" > /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/uapi/$$F; done; touch /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/uapi/.install