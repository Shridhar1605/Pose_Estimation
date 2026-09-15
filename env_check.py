"""Environment check: Python, torch, device, ONNX Runtime providers (cross-platform)."""
import json
from platform_utils import system_summary
print(json.dumps(system_summary(), indent=2))
