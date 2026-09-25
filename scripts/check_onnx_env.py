import sys

for pkg in ["onnx", "onnxruntime", "skl2onnx", "torch", "sklearn", "joblib"]:
    try:
        mod = __import__(pkg)
        print(f"✅ {pkg}: {getattr(mod, '__version__', 'installed')}")
    except ImportError as e:
        print(f"❌ {pkg}: Not installed ({e})")
