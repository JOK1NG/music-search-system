"""
强制 Python 进程的 stdout / stderr 走 UTF-8 编码。

为什么需要这个模块
==================
在 Windows PowerShell / cmd 下，Python 的 ``sys.stdout`` 默认编码会跟随系统的
ANSI 代码页（中文 Windows 通常是 ``cp936``，即 GBK）。当我们把 CLI 脚本的
print 输出重定向到日志文件 (``python evaluate_collapse.py > log.txt``) 时，
PowerShell 会用 GBK 把 Python 输出的 UTF-8 字节流再编码一次，触发以下两个连锁反应：

1. **Python 端**先用 GBK 编码 unicode 字符串，能编的中文按 GBK 字节写出，无法用
   GBK 表示的字符（如 emoji ``✅``）会被替换为 ``?``。
2. **PowerShell 端**收到 GBK 字节流时把它当成 GBK 直接落盘，但若 Python 的输出里
   混有不能被 GBK 编码而被 ``?`` 替换的位置，相邻的 ASCII 数字 / 字母可能被
   "吃掉一个"——典型症状是日志里出现 ``?994 首歌`` 这种本应是 ``7994 首歌`` 的串。

修复方式
========
在 import 副作用阶段调用 ``sys.stdout.reconfigure(encoding="utf-8", ...)``，把
当前进程的 stdout / stderr 强制改成 UTF-8 文本流。这样 Python 输出的就是合法
UTF-8，文件里再也不会被 GBK 二次编码搞坏。POSIX 平台 stdout 默认就是 UTF-8，
此调用是 no-op，因此可以安全地放在所有 CLI 脚本顶部。

用法
====
在任何会被 ``python xxx.py`` 直接执行的 CLI 脚本顶部尽早 import 一次::

    import _utf8_console  # noqa: F401

仅 import 副作用即可生效，不需要调用任何函数。
"""
from __future__ import annotations

import sys


def _enable() -> None:
    """把 stdout / stderr 切到 UTF-8。仅在解释器支持 ``reconfigure`` 时生效。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # 某些被 capsys / IDE 替换过的 stream 不接受 reconfigure，忽略即可。
                pass


_enable()
