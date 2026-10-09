"""Интерактивная лаборатория линейных моделей."""

import os

# Official ONNX Runtime builds enable telemetry during initialization. Opt out
# before imports; the API alone may run after an initialization event.
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
