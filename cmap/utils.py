import os
import shutil


def nint(value: float) -> int:
    """四捨五入でint型に変換 (Fortranのnintと同等)。
    decimal モジュールを使わず高速化。
    """
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def cd_main() -> None:
    """スクリプトの親ディレクトリに移動する。"""
    py_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(py_path, '..', '..'))
    print(f"Current directory :: {os.getcwd()}")


def copy_file(src: str, dest: str) -> None:
    """ファイルをコピーする。"""
    try:
        shutil.copy(src, dest)
    except Exception as e:
        print(f"ファイルコピー中にエラーが発生しました: {e}")
