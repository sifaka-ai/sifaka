# Transformer Backends for Sifaka Classifiers

Sifaka's text classifiers use Hugging Face Transformers, which supports multiple backends. You can choose the backend that best fits your environment and requirements.

## 🚀 Quick Start

```bash
# Default: PyTorch backend (recommended for most users)
pip install "sifaka[classifiers]"

# Alternative: TensorFlow backend
pip install "sifaka[classifiers-tf]"

# Alternative: JAX backend
pip install "sifaka[classifiers-jax]"
```

## 🔧 Backend Comparison

| Backend | Pros | Cons | Best For |
|---------|------|------|----------|
| **PyTorch** | • Most models available<br>• Best community support<br>• Excellent debugging<br>• Default for most tutorials | • Larger memory footprint<br>• Slower startup | • General use<br>• Development<br>• Research |
| **TensorFlow** | • Production optimized<br>• Good mobile/edge support<br>• TensorFlow Serving<br>• Smaller memory usage | • Fewer available models<br>• More complex debugging | • Production deployment<br>• Mobile/edge devices<br>• Serving at scale |
| **JAX** | • Fastest training<br>• Excellent for research<br>• Functional programming<br>• XLA compilation | • Smallest model selection<br>• Steeper learning curve<br>• Less mature ecosystem | • Research<br>• High-performance computing<br>• Custom training |

## 📋 Requirements

### Python Version Support
- **Python 3.9+** (supports 3.9, 3.10, 3.11, 3.12)
- All backends work with the same Python versions

### PyTorch Backend (Default)
```bash
pip install "sifaka[classifiers]"
```
**Includes:**
- `transformers>=4.52.0`
- `torch>=2.0.0`
- `sentencepiece>=0.2.0`
- `accelerate>=0.20.0`

### TensorFlow Backend
```bash
pip install "sifaka[classifiers-tf]"
```
**Includes:**
- `transformers>=4.52.0`
- `tensorflow>=2.15.0`
- `sentencepiece>=0.2.0`

### JAX Backend
```bash
pip install "sifaka[classifiers-jax]"
```
**Includes:**
- `transformers>=4.52.0`
- `jax>=0.4.0`
- `flax>=0.8.0`
- `sentencepiece>=0.2.0`

## 🎯 Choosing the Right Backend

### Use PyTorch if:
- ✅ You're new to machine learning
- ✅ You want the widest model selection
- ✅ You're doing research or experimentation
- ✅ You need extensive community support
- ✅ You're following tutorials (most use PyTorch)

### Use TensorFlow if:
- ✅ You're deploying to production
- ✅ You need mobile/edge deployment
- ✅ You want optimized serving infrastructure
- ✅ You have memory constraints
- ✅ You're using Google Cloud Platform

### Use JAX if:
- ✅ You need maximum performance
- ✅ You're doing research with custom training
- ✅ You prefer functional programming
- ✅ You're using Google TPUs
- ✅ You need XLA compilation benefits

## 🔄 Switching Backends

You can switch backends by reinstalling with a different option:

```bash
# Switch from PyTorch to TensorFlow
pip uninstall torch torchvision torchaudio
pip install "sifaka[classifiers-tf]"

# Switch from TensorFlow to JAX
pip uninstall tensorflow
pip install "sifaka[classifiers-jax]"
```

**Note**: Sifaka's classifier code is backend-agnostic. The same Python code works with any backend.

## 🐛 Troubleshooting

### Import Errors
```python
ImportError: No module named 'torch'
```
**Solution**: Install the appropriate backend:
```bash
pip install "sifaka[classifiers]"  # for PyTorch
```

### Model Loading Issues
```python
OSError: Can't load model. Repository not found.
```
**Solutions**:
1. Check internet connection
2. Verify model name is correct
3. Some models may require authentication

### Memory Issues
```python
RuntimeError: CUDA out of memory
```
**Solutions**:
1. Use CPU-only mode (default in Sifaka)
2. Switch to TensorFlow backend (more memory efficient)
3. Process shorter texts
4. Use smaller models

### Performance Issues
- **Slow startup**: Normal on first run (models download)
- **Slow inference**: Consider using cached classifiers
- **High memory usage**: Switch to TensorFlow backend

## 📊 Performance Comparison

Based on typical text classification tasks:

| Metric | PyTorch | TensorFlow | JAX |
|--------|---------|------------|-----|
| **Startup Time** | Medium | Fast | Slow |
| **Inference Speed** | Fast | Medium | Fastest |
| **Memory Usage** | High | Medium | Low |
| **Model Selection** | Excellent | Good | Limited |
| **Ease of Use** | Easy | Medium | Hard |

## 🔗 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html)

## 💡 Tips

1. **Start with PyTorch** - It's the most beginner-friendly and has the best model support
2. **Use caching** - Enable `cached=True` for better performance with any backend
3. **Monitor memory** - TensorFlow is more memory-efficient for production
4. **Consider your deployment** - TensorFlow has better production tooling
5. **Experiment** - All backends use the same Sifaka API, so switching is easy

Happy classifying! 🚀
