"""QUEST 実験データの取得インターフェース。

get_data.py の get_CHI_Data クラスを cmap パッケージから参照するための薄いラッパー。
"""

from get_data import get_CHI_Data as ExperimentalData  # noqa: F401

__all__ = ["ExperimentalData"]
