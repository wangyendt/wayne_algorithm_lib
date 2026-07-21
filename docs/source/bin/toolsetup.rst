toolsetup: 跨平台开发环境配置
================================

``toolsetup`` 是安装 ``pywayne`` 后提供的环境配置工具。它使用统一的
``--task`` 和 ``--platform`` 参数管理 shell 快捷命令、Node.js/npm、
Tailscale、MongoDB 和 PyMongo 环境。

工具会在执行安装前显示命令。安装类任务默认要求确认，自动化场景可使用
``--yes``；建议第一次运行时先加 ``--dry-run`` 检查计划。

支持范围
--------

.. list-table::
   :header-rows: 1

   * - 任务
     - macOS
     - Linux
     - Windows
   * - ``shortcuts``
     - zsh / bash
     - bash / zsh，包含 ``gpu``
     - PowerShell
   * - ``npm``
     - 支持
     - 支持
     - 不支持
   * - ``tailscale``
     - 不支持
     - 支持
     - 不支持
   * - ``mongodb``
     - 支持
     - 支持
     - 不支持

Windows 当前只配置 PowerShell 快捷命令，可通过随包安装的
``toolsetup.cmd`` 或 ``toolsetup.py`` 运行。按设计，``gpu`` 命令只会在
Linux 加载后可用；macOS 和 Windows 不会提供该命令。

快速开始
--------

.. code-block:: bash

   # 自动识别当前平台，预览全部适用任务
   toolsetup --task all --platform auto --dry-run

   # 配置当前平台的 shell 快捷命令
   toolsetup --task shortcuts --platform auto

   # Linux 安装全部支持的任务，不再询问确认
   toolsetup --task all --platform linux --yes

   # 在 macOS 安装 Node.js/npm
   toolsetup --task npm --platform macos --yes

共同参数
--------

- ``--task {shortcuts,npm,tailscale,mongodb,all}``：选择任务，必填。
- ``--platform {auto,macos,linux,windows}``：目标平台，默认自动识别。
- ``--shell {auto,bash,zsh,powershell}``：指定要配置的 shell。
- ``--dry-run``：只显示配置文件和命令计划，不修改系统。
- ``--yes``：跳过 npm、Tailscale、MongoDB 等安装任务的确认提示。
- ``--force``：替换已有的受管配置块，或重新运行已经安装过的 nvm 安装器。

显式指定的目标平台与当前主机不一致时，只允许 ``--dry-run``，防止在
错误系统上执行 ``apt``、``brew`` 或 PowerShell 配置。

Shell 快捷命令
--------------

.. code-block:: bash

   toolsetup --task shortcuts --platform linux --shell bash
   toolsetup --task shortcuts --platform macos --shell zsh
   toolsetup --task shortcuts --platform windows

Unix 默认写入 ``~/.bashrc`` 或 ``~/.zshrc``；Windows 写入当前用户的
PowerShell Profile。也可以用 ``--rc-file`` 指定其他文件：

.. code-block:: bash

   toolsetup --task shortcuts --platform linux --rc-file /tmp/test.bashrc

配置以如下标记包围：

.. code-block:: text

   # >>> pywayne toolsetup shortcuts >>>
   ...
   # <<< pywayne toolsetup shortcuts <<<

如果标记已经存在，默认跳过，不会产生重复函数。使用 ``--force`` 时替换
受管块，并先生成形如 ``.zshrc.bak.20260720123000123456`` 的备份。

代理命令
~~~~~~~~

``proxy_on`` 设置大小写两套 HTTP、HTTPS、SOCKS 和 NO_PROXY 环境变量，
同时写入 Git 全局代理；``proxy_off`` 清理这些变量和 Git 配置。

.. code-block:: bash

   toolsetup --task shortcuts --platform auto \
     --http-proxy http://127.0.0.1:7890 \
     --socks-proxy socks5://127.0.0.1:7890 \
     --no-proxy 'localhost,127.0.0.1,.example.com'

修改 rc 文件后，需要重新打开终端，或按工具输出执行 ``source``。代理环境
变量只影响加载配置后的 shell；Git 代理使用 ``git config --global``，会影响
当前用户的所有 Git 仓库。

goto 路径跳转
~~~~~~~~~~~~~

``goto`` 将快捷名持久化到 Unix 的 ``~/.goto_paths`` 或 Windows 的
``~/.goto_paths.json``：

.. code-block:: bash

   goto add code ~/Documents/work/code
   goto add project "/path/with spaces/project"
   goto list
   goto code
   goto remove project
   goto clear

安装快捷命令时可以同时初始化路径；``--goto`` 可重复：

.. code-block:: bash

   toolsetup --task shortcuts --platform auto \
     --goto code="$HOME/Documents/work/code" \
     --goto tools="$HOME/Documents/tools"

快捷名仅允许字母、数字、点、下划线和连字符，初始化时目标目录必须已经存在。

Linux GPU 概览
~~~~~~~~~~~~~~

``gpu`` 使用 ``nvidia-smi`` 显示以下内容：

- GPU 型号、温度、风扇、功率、显存和利用率；
- 每张 GPU 的 compute 进程、PID、用户、显存、运行时长和命令；
- 按用户汇总的 compute 进程显存占用。

它依赖 NVIDIA 驱动与 ``nvidia-smi``。图形进程或驱动保留显存可能不会出现
在 compute 进程列表中。

Node.js 与 npm
---------------------

.. code-block:: bash

   # 默认安装 nvm v0.40.6、Node.js 20，并设置 npmmirror
   toolsetup --task npm --platform linux --yes

   # macOS
   toolsetup --task npm --platform macos --yes

   # 改用其他 Node 版本或 npm registry
   toolsetup --task npm --platform auto --node-version 24 \
     --npm-registry https://registry.npmjs.org --yes

Linux 会根据系统选择 apt、dnf、yum、zypper 或 apk 安装 curl、Git 和编译
工具。macOS 需要预先安装 Xcode Command Line Tools；nvm 官方不支持通过
Homebrew 安装，因此 macOS 和 Linux 都使用 nvm 官方安装脚本。

随后工具会列出远端 LTS 版本，安装 ``--node-version``，设置为默认版本，
并通过 ``npm config set registry`` 保存 registry。默认值保持为需求中的
Node.js 20；可以显式改为 22、24 或 ``lts/*``。

Linux Tailscale
---------------

.. code-block:: bash

   toolsetup --task tailscale --platform linux --dry-run
   toolsetup --task tailscale --platform linux --yes

该任务下载并执行 Tailscale 官方 Linux 安装脚本，启用 ``tailscaled`` 服务，
然后运行 ``sudo tailscale up``。首次连接通常会在终端打印认证 URL，需要在
浏览器完成登录。

MongoDB 与 PyMongo
-------------------------

.. code-block:: bash

   # macOS：Homebrew 安装并启动 MongoDB 服务
   toolsetup --task mongodb --platform macos --yes

   # Linux：官方仓库安装，systemd 设置开机启动
   toolsetup --task mongodb --platform linux --yes

   # 自定义 PyMongo 虚拟环境
   toolsetup --task mongodb --platform auto \
     --pymongo-venv ~/.venvs/my-mongodb --yes

   # 只安装 MongoDB 服务
   toolsetup --task mongodb --platform auto --skip-pymongo --yes

当前经过验证的服务器版本是 MongoDB 8.0：

- macOS 通过 ``mongodb/brew`` 安装 ``mongodb-community@8.0``，并执行
  ``brew services start``，服务会随用户登录启动；
- Ubuntu 20.04、22.04、24.04 和 Debian 12 使用 MongoDB 官方 apt 仓库；
- RHEL、Rocky Linux、AlmaLinux、CentOS Stream 和 Oracle Linux 8/9 使用
  MongoDB 官方 yum/dnf 仓库；
- Linux 使用 ``systemctl enable --now mongod`` 启动并设置开机自启。

PyMongo 默认安装到独立的 ``~/.venvs/pywayne-mongodb``，避免修改系统
Python。激活方式：

.. code-block:: bash

   source ~/.venvs/pywayne-mongodb/bin/activate
   python -c "import pymongo; print(pymongo.version)"

验证与排错
----------

.. code-block:: bash

   # 检查生成计划
   toolsetup --task all --platform auto --dry-run

   # MongoDB 服务状态
   systemctl status mongod             # Linux
   brew services list                  # macOS

   # npm 配置
   node -v
   npm -v
   npm config get registry

   # Tailscale 状态
   tailscale status

Ubuntu/Debian 自带的旧 ``mongodb`` 包会与官方 ``mongodb-org`` 冲突。
``toolsetup`` 检测到该包时会停止，不会自动删除软件或数据；请先自行审查现有
数据库与备份，再决定是否卸载。

安全说明
--------

- npm/nvm 和 Tailscale 任务会从官方站点下载脚本并执行；先用
  ``--dry-run`` 核对 URL 和命令。
- Linux 安装会使用 ``sudo``；工具不会自动删除冲突包或数据库文件。
- ``proxy_on`` 会修改 Git 全局配置，使用 ``proxy_off`` 可恢复。
- MongoDB 默认配置适合本机开发。对外开放端口前，应单独配置认证、TLS、
  防火墙和备份策略。

上游安装依据
------------

- `nvm 官方安装说明 <https://github.com/nvm-sh/nvm#installing-and-updating>`_
- `Tailscale Linux 安装说明 <https://tailscale.com/docs/install/linux>`_
- `MongoDB 8.0 Ubuntu 安装说明 <https://www.mongodb.com/docs/v8.0/tutorial/install-mongodb-on-ubuntu/>`_
- `MongoDB 8.0 Debian 安装说明 <https://www.mongodb.com/docs/v8.0/tutorial/install-mongodb-on-debian/>`_
- `MongoDB 8.0 RHEL 安装说明 <https://www.mongodb.com/docs/v8.0/tutorial/install-mongodb-on-red-hat/>`_
- `MongoDB 8.0 macOS 安装说明 <https://www.mongodb.com/docs/v8.0/tutorial/install-mongodb-on-os-x/>`_
