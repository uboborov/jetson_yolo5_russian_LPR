cmd_/home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/scsi/.install := /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/scsi ./include/uapi/scsi cxlflash_ioctl.h scsi_bsg_fc.h scsi_netlink.h scsi_netlink_fc.h; /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/scsi ./include/scsi ; /bin/bash scripts/headers_install.sh /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/scsi ./include/generated/uapi/scsi ; for F in ; do echo "$(pound)include <asm-generic/$$F>" > /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/scsi/$$F; done; touch /home/ubobrov/develop/projects/nvidia/jetson_nano/projects/tegra/include/scsi/.install