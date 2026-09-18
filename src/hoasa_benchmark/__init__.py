"""HOASA ABSA Indonesian hotel-review benchmark for small language models.

Public surface is the `hoasa` CLI (see `hoasa_benchmark.cli`). Heavy ML imports
are lazy inside the baseline/train/evaluate/preflight functions so that
prepare/majority/CLI --help work in a stdlib + PyYAML environment.
"""

__version__ = "0.1.0"

PACKAGE_NAME = "hoasa-slm-benchmark"

__all__ = ["__version__", "PACKAGE_NAME"]
