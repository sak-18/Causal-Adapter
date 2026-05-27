# Environment Files

Place environment definitions here for reproducible setup.

Recommended files:

- causal-adapter-sd15.yml
- causal-adapter-sd3.yml
- benchmark.yml

## Usage

Run commands from repository root.

### SD1.5

```bash
conda env create -f envs/causal-adapter-sd15.yml
conda activate causal-adapter-sd15
pip install -e causal-adapter-sd15/diffusers
```

### SD3 / Flux

```bash
conda env create -f envs/causal-adapter-sd3.yml
conda activate causal-adapter-sd3
pip install -e causal-adapter-sd3/diffusers
```

### Benchmark

```bash
conda env create -f envs/benchmark.yml
conda activate causal-adapter-benchmark
```
