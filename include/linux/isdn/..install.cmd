cmd_/home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/linux/isdn/.install := /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/linux/isdn ./include/uapi/linux/isdn capicmd.h; /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/linux/isdn ./include/linux/isdn ; /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/linux/isdn ./include/generated/uapi/linux/isdn ; for F in ; do echo "$(pound)include <asm-generic/$$F>" > /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/linux/isdn/$$F; done; touch /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/linux/isdn/.install