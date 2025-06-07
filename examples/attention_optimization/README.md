# MLIR Attention Optimization with OpenEvolve

This example demonstrates using OpenEvolve to optimize attention mechanisms at the MLIR IR level, following the approach described in DeepMind's AlphaEvolve paper.

## Overview

- **Goal**: Achieve 15-32% speedup in attention implementations
- **Method**: Evolutionary optimization of MLIR transformation parameters
- **Target**: Match AlphaEvolve's compiler optimization results

## Quick Start

1. **Setup environment** (from openevolve root):
   ```bash
   source .venv/bin/activate
   export OPENAI_API_KEY='your-key-here'
   ```

2. **Run optimization**:
   ```bash
   cd examples/attention_optimization
   python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 100
   ```

3. **Monitor progress**:
   ```bash
   # Watch for checkpoint updates
   ls -la openevolve_output/checkpoints/
   
   # View best results
   cat openevolve_output/checkpoints/checkpoint_*/best_program_info.json
   ```

## How It Works

### 1. Initial Program (`initial_program.py`)
Defines baseline optimization parameters:
- Tiling strategies
- Memory layout options  
- Vectorization settings
- Fusion configurations

### 2. Evaluator (`evaluator.py`)
Evaluates optimization effectiveness:
- Applies MLIR transformations
- Benchmarks performance
- Calculates speedup vs baseline
- Returns score for OpenEvolve

### 3. Evolution Process
OpenEvolve iteratively:
- Generates new parameter combinations
- Evaluates each variant
- Selects best performers
- Mutates and evolves solutions

## Expected Results

Based on AlphaEvolve paper:
- **Iteration 1-50**: 5-10% improvements from basic optimizations
- **Iteration 50-200**: 15-20% improvements from evolved patterns
- **Iteration 200+**: 25-32% target speedup achieved

## Files Structure

```
attention_optimization/
├── initial_program.py      # Starting optimization program
├── evaluator.py           # Evaluation script  
├── config.yaml           # OpenEvolve configuration
├── mlir/
│   └── baseline_attention.mlir  # Base MLIR implementation
├── results/               # Evolution outputs
└── checkpoints/          # Evolution checkpoints
```

## Customization

### Modify Optimization Space
Edit `initial_program.py` to change:
- Tile size ranges
- Available transformations
- Hardware-specific options

### Adjust Evolution Parameters  
Edit `config.yaml` to modify:
- Population size
- Iteration count
- Model selection
- Convergence criteria

### Extend Evaluation
Edit `evaluator.py` to add:
- More test configurations
- Additional metrics
- Hardware-specific benchmarks

## Research Applications

This framework enables research into:
- Automated compiler optimization
- Attention mechanism efficiency
- Multi-level IR optimization
- Evolutionary programming techniques

## Next Steps

1. **Validate results** with real MLIR compilation
2. **Extend to GPU/TPU** targets  
3. **Add more operations** (convolution, etc.)
4. **Scale to larger models** and datasets