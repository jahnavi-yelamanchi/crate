"""Quantize the CLAP audio encoder → int8 ONNX, and benchmark latency.

The audio tower is the hot path (every clip at index time, every hum/track at
query time). We export it, dynamic-int8 quantize, and time base-torch vs int8-onnx
so the latency dashboard has real numbers, not vibes.

    python -m crate.model.quantize            # export + quantize + bench
"""

from __future__ import annotations

import json
import time

import numpy as np

from crate import config


def export_and_quantize(adapter: str | None = None) -> str:
    """Export merged audio encoder to ONNX, then dynamic int8. Returns onnx path."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from crate.model.encoder import ClapEncoder

    enc = ClapEncoder(adapter=adapter, device="cpu")
    processor = enc.processor
    dummy = np.zeros(config.CLIP_SAMPLES, dtype=np.float32)
    feats = processor(audios=[dummy], sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    input_features = feats["input_features"]

    class AudioTower(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features):
            return self.model.get_audio_features(input_features=input_features)

    tower = AudioTower(enc.model).eval()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fp32 = str(config.ONNX_PATH).replace(".int8", ".fp32")
    torch.onnx.export(
        tower, (input_features,), fp32,
        input_names=["input_features"], output_names=["embedding"],
        dynamic_axes={"input_features": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    quantize_dynamic(fp32, str(config.ONNX_PATH), weight_type=QuantType.QInt8)
    print(f"quantized encoder → {config.ONNX_PATH}")
    return str(config.ONNX_PATH)


def benchmark(n: int = 50, adapter: str | None = None) -> dict:
    """Median per-clip latency: base torch (CPU) vs int8 ONNX. Writes eval/latency.json."""
    import onnxruntime as ort

    from crate.model.encoder import ClapEncoder

    enc = ClapEncoder(adapter=adapter, device="cpu")
    processor = enc.processor
    clip = np.random.randn(config.CLIP_SAMPLES).astype(np.float32)
    feats = processor(audios=[clip], sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    onnx_in = {"input_features": feats["input_features"].numpy()}

    base_ms = _median_ms(lambda: enc.embed_audio(clip), n)

    sess = ort.InferenceSession(str(config.ONNX_PATH), providers=["CPUExecutionProvider"])
    int8_ms = _median_ms(lambda: sess.run(None, onnx_in), n)

    result = {
        "base_torch_cpu_ms": round(base_ms, 2),
        "int8_onnx_cpu_ms": round(int8_ms, 2),
        "speedup": round(base_ms / int8_ms, 2) if int8_ms else None,
        "clip_seconds": config.CLIP_SECONDS,
    }
    config.EVAL_DIR.mkdir(exist_ok=True)
    (config.EVAL_DIR / "latency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _median_ms(fn, n: int) -> float:
    fn()  # warm up
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


if __name__ == "__main__":
    export_and_quantize()
    benchmark()
