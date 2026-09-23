"""指数缠论分析扩展 — 解耦的 L2 二开模块。

对指数日 K / 分钟 K 做多级别缠论结构分析 (笔/中枢), 前端指数页调用。
czsc 随主依赖安装。导入失败时回退内置简化分型, 不做包含处理。
删除本目录即可整体卸载, 不影响核心功能; 详见 docs/secondary-development.md。
"""
from __future__ import annotations

from app.extensions import (
    BACKEND_EXTENSION_API_VERSION,
    BackendExtensionRegistrar,
)

EXTENSION_ID = "indices.chan"
EXTENSION_API_VERSION = BACKEND_EXTENSION_API_VERSION


def setup(registrar: BackendExtensionRegistrar) -> None:
    from app.custom.chan.routes import build_router

    registrar.include_router(build_router())
