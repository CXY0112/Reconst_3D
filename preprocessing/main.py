"""无参数启动点云工作台，默认进入预处理阶段。"""
from pathlib import Path
import sys

# Resolve imports relative to this file, including when launched from an IDE.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_workbench import launch


if __name__ == '__main__':
    launch(initial_stage='preprocessing')
