cmd_/home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/misc/.install := /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/misc ./include/uapi/misc cxl.h; /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/misc ./include/misc ; /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/misc ./include/generated/uapi/misc ; for F in ; do echo "$(pound)include <asm-generic/$$F>" > /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/misc/$$F; done; touch /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/misc/.install
