# 相机部署

## Windows x64

运行使用 vendor/camera_win64/AngstrongCameraSdk.dll 和 configurationfiles/hp60c_v2_00_20230704_configEncrypt.json；camera_win64/CONTENTS.json 保留原资源来源。采用已验证 Viewer v1.2.20 SDK，不以其他 SDK 版本直接替换。

本地资源中未发现单独的该相机 INF/SYS 安装包；Windows 使用设备的 UVC 系统驱动。CP210x 是串口驱动，不是本相机驱动，没有打进相机包。目标机若设备管理器异常，请使用厂商针对该设备提供的驱动，不能将 SDK DLL 当作内核驱动安装。

设置 RECONST_CAMERA_SDK 可指定其他 SDK 目录。RGB/深度均为 640×480，启动后等待1秒再正式采样。实际 USB 访问、固件、标定和首帧图像需目标机实测。

## Linux

迁移包 vendor/camera_linux 包含从现有厂商 linux_ros.pkg 安全展开的 libs/include、libs/lib/{x86_64-linux-gnu,aarch64-linux-gnu,arm-linux-gnueabihf}、scripts/angstrong-camera.rules 和厂商 linux 示例。配置位于 configurationfiles。

先检查目标架构与共享库依赖，再审阅并按目标系统安装 udev 规则、重载规则及重新插拔。没有替用户运行驱动安装脚本，也没有更改本机 USB 权限。主项目 camera_sdk.py 中 Windows ctypes 绑定尚需转换为 Linux SDK API/ABI；不能仅复制 .so 就声称主工作台已适配完成。

SDK 原始资料见 vendor/camera_docs（仅迁移包）。不得将实拍中的私有标定、USB序列号或历史采集作为公共默认配置。
