"""基于 RGB-D 的三维场景重建与交互建模；几何重建与独立物体建模，beta 为精简演示流程。"""
from desktop_display import enable_dpi_awareness
enable_dpi_awareness()
from scene_workbench import launch


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',choices=('beta','rgbd'),help='beta:精简流程；rgbd:兼容旧启动方式')
    args=parser.parse_args(argv)
    launch(beta=args.mode=='beta')


if __name__=='__main__':main()
