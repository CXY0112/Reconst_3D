"""Run retained regressions from any working directory; no camera required."""
from pathlib import Path
import sys
import unittest
import gc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class MainThreadCleanupResult(unittest.TextTestResult):
    def stopTest(self, test):
        super().stopTest(test)
        # Repeated Tk roots leave widget/reference cycles after destroy(). Reclaim
        # them here, before the next test starts workers that could trigger GC.
        gc.collect()


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2, resultclass=MainThreadCleanupResult).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
