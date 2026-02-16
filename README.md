# Deep-Learning-Inference-Optimization-on-GPU


## 1. Project Overview

This project investigates and benchmarks deep learning inference performance on GPU, comparing **PyTorch eager execution** with **ONNX Runtime** using the CUDA Execution Provider.

The primary objective is to analyze:

* Latency
* Throughput
* Batch scaling behavior
* GPU memory utilization
* Graph optimization impact

All experiments were conducted on **NVIDIA Tesla P100 (16GB HBM2)** in a Kaggle GPU environment.

---

## 2. Motivation

In production systems, inference performance directly impacts:

* API response time
* Server cost
* Scalability
* Real-time capability

While PyTorch is commonly used for model training, optimized inference engines such as **ONNX Runtime** provide:

* Static graph execution
* Kernel fusion
* Reduced runtime overhead
* Hardware-specific optimizations

This project evaluates the practical performance gain of exporting models to ONNX and deploying with GPU acceleration.

---

## 3. Experimental Setup

### Hardware

* GPU: NVIDIA Tesla P100
* VRAM: 16GB
* Architecture: Pascal (No Tensor Cores)

### Software

* PyTorch
* ONNX
* ONNX Runtime (CUDAExecutionProvider)
* Python 3.x

### Inference Mode

* FP32 precision
* No gradient computation
* Warm-up iterations before timing
* Averaged over multiple runs

---

## 4. Benchmark Results

### Latency Comparison (Batch = 1)

| Framework    | Latency (ms) |
| ------------ | ------------ |
| PyTorch      | 6.51 ms      |
| ONNX Runtime | 5.18 ms      |

ONNX Runtime achieved approximately **~20% latency reduction** compared to PyTorch eager execution.

---

### Batch Scaling Behavior

| Batch Size | Latency (ms) |
| ---------- | ------------ |
| 1          | 5.36         |
| 4          | 9.82         |
| 8          | 16.32        |
| 16         | 28.85        |

Throughput increases with batch size, demonstrating effective GPU parallelization.

Scaling is sub-linear, indicating efficient utilization of CUDA cores and memory bandwidth.

---

### GPU Memory Usage

Allocated GPU memory during inference:

```
~106 MB
```

Memory footprint includes:

* Model weights
* Activation buffers
* CUDA context
* Runtime workspace

No abnormal memory growth was observed.

---

## 5. Key Findings

1. ONNX Runtime reduces inference latency due to static graph optimization.
2. GPU utilization improves significantly as batch size increases.
3. Batch size presents a trade-off between latency and throughput.
4. Tesla P100 (Pascal) performs best with FP32, as it does not contain Tensor Cores.
5. CUDAExecutionProvider is sufficient for high-performance inference on this hardware.

---

## 6. Technical Insights

### Why ONNX Runtime is Faster

* Static computation graph
* Operator fusion
* Reduced Python overhead
* Optimized CUDA kernel dispatch

Unlike PyTorch eager mode, ONNX Runtime avoids dynamic graph interpretation overhead.



---

## 7. Future Improvements

Potential extensions of this project:

* TensorRT Execution Provider benchmarking
* FP16 mixed precision inference
* INT8 quantization experiments
* Cold-start latency measurement
* Nsight Systems profiling
* Dynamic input shape benchmarking

---

## 8. Conclusion

This project demonstrates that exporting PyTorch models to ONNX and deploying with ONNX Runtime can yield measurable performance improvements in GPU inference.

The results highlight the importance of:

* Proper benchmarking methodology
* Understanding hardware characteristics
* Evaluating latency–throughput trade-offs
* Choosing the correct execution backend for deployment

This work reflects a production-oriented approach to deep learning model optimization.

---
