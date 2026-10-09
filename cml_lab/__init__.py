"""CML-lab package. Numerical workers run without runtime telemetry."""
import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
__version__ = "4.0.0"
