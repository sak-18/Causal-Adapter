# Environment Files

Place environment definitions here for reproducible setup.

Recommended files:

- causal-adapter-sd15.yml
- causal-adapter-sd3.yml
- benchmark.yml

Environment mapping used in this project:

- `flux`: SD3 / Flux
- `mcpl`: SD1.5 + benchmark

## Usage

Run commands from repository root.

### SD1.5

```bash
conda env create -f envs/causal-adapter-sd15.yml
conda activate mcpl
pip install -e causal-adapter-sd15/diffusers
```

### SD3 / Flux

```bash
conda env create -f envs/causal-adapter-sd3.yml
conda activate flux
pip install -e causal-adapter-sd3/diffusers
```

### Benchmark

```bash
conda activate mcpl
```

If `mcpl` does not exist yet, create it with:

```bash
conda env create -f envs/causal-adapter-sd15.yml
```
