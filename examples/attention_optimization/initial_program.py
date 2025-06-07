#!/usr/bin/env python3
"""
Initial attention optimization program for AlphaEvolve reproduction.
This program defines MLIR transformation parameters that will be evolved.
Targets 32% speedup like the original AlphaEvolve paper.
"""

import json
import sys
import random

def optimize_attention():
    """
    Define attention optimization parameters for evolution.
    
    The goal is to achieve 32% speedup (1.32x) like AlphaEvolve paper
    by optimizing compiler-generated MLIR IR for attention kernels.
    """
    
    # AlphaEvolve-inspired parameter space exploration
    # These parameters control MLIR compiler transformations
    
    # Memory tiling strategy - crucial for cache performance  
    # Based on typical L1/L2 cache sizes and attention patterns
    tile_options_m = [16, 32, 64, 128]  # Sequence dimension tiles
    tile_options_n = [32, 64, 128, 256] # Head dimension tiles
    
    # Smart initialization: favor cache-friendly sizes
    tile_size_m = random.choice([32, 64])  # Sweet spot for most caches
    tile_size_n = random.choice([64, 128]) # Head dim optimization
    
    # Vectorization strategy - critical for modern SIMD
    vectorization_options = ['none', 'affine', 'linalg']
    vectorization = random.choice(vectorization_options)
    
    # Loop unrolling - balance code size vs performance
    unroll_factors = [1, 2, 4, 8]
    # Favor moderate unrolling for attention kernels
    unroll_factor = random.choice([2, 4] if random.random() > 0.5 else unroll_factors)
    
    # Fusion strategy - key for reducing memory traffic
    fusion_strategies = ['none', 'producer', 'consumer', 'both']
    # Favor fusion for attention (Q@K^T, softmax, @V pattern)
    fusion_strategy = random.choice(['both', 'producer'] if random.random() > 0.3 else fusion_strategies)
    
    # Loop interchange - can improve memory access patterns
    loop_interchange = random.choice([True, False])
    
    # Memory optimizations - crucial for large attention matrices
    use_shared_memory = random.choice([True, False])
    
    # Performance vs latency trade-off
    optimize_for_latency = random.choice([True, False])
    
    # Additional optimizations inspired by FlashAttention
    enable_blocking = random.choice([True, False])  # Block-wise computation
    enable_recomputation = random.choice([True, False])  # Memory vs compute trade-off
    
    optimization_params = {
        # Core tiling parameters
        'tile_size_m': tile_size_m,
        'tile_size_n': tile_size_n,
        
        # Vectorization and parallelization
        'vectorization': vectorization,
        'unroll_factor': unroll_factor,
        'loop_interchange': loop_interchange,
        
        # Fusion and memory optimization
        'fusion_strategy': fusion_strategy,
        'use_shared_memory': use_shared_memory,
        
        # Performance tuning
        'optimize_for_latency': optimize_for_latency,
        'enable_blocking': enable_blocking,
        'enable_recomputation': enable_recomputation,
        
        # Metadata for analysis
        'optimization_strategy': 'alphaevolve_inspired',
        'target_speedup': 1.32,
    }
    
    return optimization_params

if __name__ == "__main__":
    # Test the function
    params = optimize_attention()
    print(json.dumps(params, indent=2))


# """
# Fixed MLIR compiler integration for attention optimization.
# Uses correct mlir-opt syntax and optimizations for FlashAttention-style kernels.
# Reproduces AlphaEvolve's approach for optimizing compiler-generated IR.
# """

# import sys
# import json
# import subprocess
# import tempfile
# import time
# import os
# import shlex
# from pathlib import Path

# class FixedMLIRCompiler:
#     """Fixed MLIR compilation with correct pass syntax"""
    
#     def __init__(self, mlir_opt_path="mlir-opt", mlir_translate_path="mlir-translate"):
#         self.mlir_opt = mlir_opt_path
#         self.mlir_translate = mlir_translate_path
#         self.temp_dir = Path(tempfile.mkdtemp(prefix="mlir_attention_"))
        
#         # Verify MLIR tools are available
#         self.verify_mlir_tools()
    
#     def verify_mlir_tools(self):
#         """Verify MLIR tools are available and working"""
#         try:
#             # Test mlir-opt
#             result = subprocess.run([self.mlir_opt, "--version"], 
#                                   capture_output=True, text=True, timeout=10)
#             if result.returncode != 0:
#                 raise RuntimeError(f"mlir-opt not working: {result.stderr}")
            
#             print(f"✅ MLIR tools verified: {self.mlir_opt}")
            
#         except FileNotFoundError as e:
#             raise RuntimeError(f"MLIR tools not found in PATH. Please add MLIR bin directory to PATH.")
#         except Exception as e:
#             raise RuntimeError(f"MLIR tools verification failed: {e}")
    
#     def compile_mlir(self, mlir_code, passes=None):
#         """Compile MLIR code with real mlir-opt using correct syntax"""
#         try:
#             # Write MLIR to temporary file
#             mlir_file = self.temp_dir / "input.mlir"
#             with open(mlir_file, 'w') as f:
#                 f.write(mlir_code)
            
#             # Build pass pipeline using correct syntax
#             if passes:
#                 # Use pass pipeline syntax for modern MLIR
#                 if isinstance(passes, list):
#                     # Convert list of passes to pipeline format
#                     pass_str = ",".join(passes)
#                     cmd = [self.mlir_opt, str(mlir_file), f"--pass-pipeline=builtin.module({pass_str})"]
#                 else:
#                     cmd = [self.mlir_opt, str(mlir_file), f"--pass-pipeline={passes}"]
#             else:
#                 # Default passes for basic optimization
#                 cmd = [self.mlir_opt, str(mlir_file), 
#                        "--canonicalize", 
#                        "--cse",
#                        "--symbol-dce"]
            
#             # Run compilation
#             result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
#             if result.returncode != 0:
#                 return None, result.stderr
            
#             return result.stdout, None
            
#         except subprocess.TimeoutExpired:
#             return None, "MLIR compilation timed out"
#         except Exception as e:
#             return None, f"MLIR compilation error: {e}"
    
#     def apply_transform_passes(self, mlir_code, transform_params):
#         """Apply transformation passes using correct MLIR syntax"""
        
#         passes = []
        
#         # Basic cleanup passes (always include)
#         passes.extend(["canonicalize", "cse"])
        
#         # Tiling passes (using correct linalg pass syntax)
#         tile_size_m = transform_params.get('tile_size_m', 0)
#         tile_size_n = transform_params.get('tile_size_n', 0)
#         if tile_size_m > 1 and tile_size_n > 1:
#             # Use the correct linalg-tile syntax from MLIR help
#             passes.append(f"linalg-tile{{linalg-tile-sizes={tile_size_m},{tile_size_n}}}")
        
#         # Vectorization passes
#         vectorization = transform_params.get('vectorization', 'none')
#         if vectorization != 'none':
#             if vectorization == 'affine':
#                 passes.append("affine-vectorize")
#             elif vectorization == 'linalg':
#                 passes.append("convert-linalg-to-vector")
        
#         # Loop unrolling 
#         unroll_factor = transform_params.get('unroll_factor', 1)
#         if unroll_factor > 1:
#             passes.append(f"affine-loop-unroll{{unroll-factor={unroll_factor}}}")
        
#         # Fusion passes
#         fusion_strategy = transform_params.get('fusion_strategy', 'none')
#         if fusion_strategy != 'none':
#             passes.append("linalg-fusion")
        
#         # Loop interchange
#         if transform_params.get('loop_interchange', False):
#             # Note: loop interchange in MLIR is often done through tiling strategies
#             passes.append("canonicalize")  # Canonicalization can reorder some operations
        
#         # Memory optimization
#         if transform_params.get('use_shared_memory', False):
#             passes.append("linalg-promote-subviews")
        
#         # Final cleanup
#         passes.extend(["canonicalize", "cse", "symbol-dce"])
        
#         # Create pipeline string
#         pipeline = f"builtin.module({','.join(passes)})"
        
#         return self.compile_mlir(mlir_code, pipeline)

#     def benchmark_mlir(self, optimized_mlir, test_config):
#         """Benchmark MLIR implementation using compilation time and IR complexity"""
        
#         try:
#             batch, heads, seq_len, head_dim = test_config
            
#             # Write optimized MLIR to file
#             benchmark_file = self.temp_dir / f"benchmark_{batch}_{heads}_{seq_len}_{head_dim}.mlir"
#             with open(benchmark_file, 'w') as f:
#                 f.write(optimized_mlir)
            
#             # Measure compilation time for lowering pipeline
#             start_time = time.time()
            
#             # Use correct lowering pipeline syntax with updated pass names
#             lowering_pipeline = (
#                 "builtin.module("
#                 "canonicalize,"
#                 "cse,"
#                 "symbol-dce,"
#                 "convert-linalg-to-loops,"
#                 "lower-affine,"
#                 "convert-scf-to-cf,"
#                 "convert-cf-to-llvm,"
#                 "convert-func-to-llvm,"
#                 "reconcile-unrealized-casts"
#                 ")"
#             )
            
#             cmd = [self.mlir_opt, str(benchmark_file), f"--pass-pipeline={lowering_pipeline}"]
            
#             result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
#             compilation_time = time.time() - start_time
            
#             if result.returncode != 0:
#                 # Compilation failed - but measure what we can
#                 ir_lines = len(optimized_mlir.split('\n'))
#                 complexity_penalty = 2.0  # Penalty for compilation failure
#                 estimated_runtime = compilation_time * complexity_penalty * (ir_lines / 50)
#                 return estimated_runtime, f"Lowering failed: {result.stderr[:200]}"
            
#             # Measure IR complexity
#             ir_lines = len(result.stdout.split('\n'))
            
#             # Calculate performance metric (lower is better)
#             # This simulates runtime based on compilation characteristics
#             base_complexity = 50
#             complexity_factor = ir_lines / base_complexity
#             time_factor = compilation_time * 2  # Weight compilation time
            
#             estimated_runtime = complexity_factor + time_factor
            
#             # Scale by workload size
#             workload_scale = (batch * heads * seq_len * head_dim) / (1 * 8 * 128 * 64)
#             estimated_runtime *= workload_scale
            
#             return estimated_runtime, None
            
#         except subprocess.TimeoutExpired:
#             return 10.0, "Compilation timeout"
#         except Exception as e:
#             return 10.0, f"Benchmark error: {e}"

# class FixedMLIRAttentionEvaluator:
#     """Evaluates MLIR attention optimizations using corrected MLIR compiler"""
    
#     def __init__(self):
#         # Initialize fixed MLIR compiler
#         self.compiler = FixedMLIRCompiler()
        
#         # Load base MLIR implementation
#         self.reference_performance = None
        
#         # Test configurations (representing different attention sizes)
#         self.test_configs = [
#             (1, 8, 128, 64),   # Small: typical inference
#             (2, 12, 256, 64),  # Medium: larger model
#             (1, 16, 512, 64),  # Large: very large sequence
#         ]
    
#     def create_baseline_mlir(self):
#         """Create a realistic baseline MLIR attention implementation"""
#         baseline = '''
# #map_q = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>
# #map_k = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
# #map_scores = affine_map<(b, h, s1, s2, d) -> (b, h, s1, s2)>
# #map_weights = affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>
# #map_v = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
# #map_out = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>

# module {
#   func.func @baseline_attention(
#       %query: tensor<1x8x128x64xf32>,
#       %key: tensor<1x8x128x64xf32>, 
#       %value: tensor<1x8x128x64xf32>
#   ) -> tensor<1x8x128x64xf32> {
    
#     %c0 = arith.constant 0.0 : f32
#     %cst_scale = arith.constant 0.125 : f32
    
#     // Initialize output tensors
#     %scores_init = tensor.empty() : tensor<1x8x128x128xf32>
#     %output_init = tensor.empty() : tensor<1x8x128x64xf32>
    
#     // Compute Q @ K^T (scaled dot-product attention)
#     %attention_scores = linalg.generic {
#       indexing_maps = [#map_q, #map_k, #map_scores],
#       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
#     } ins(%query, %key : tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
#       outs(%scores_init : tensor<1x8x128x128xf32>) {
#     ^bb0(%q: f32, %k: f32, %acc: f32):
#       %prod = arith.mulf %q, %k : f32
#       %scaled = arith.mulf %prod, %cst_scale : f32
#       %sum = arith.addf %acc, %scaled : f32
#       linalg.yield %sum : f32
#     } -> tensor<1x8x128x128xf32>
    
#     // Apply attention weights to values  
#     %attention_output = linalg.generic {
#       indexing_maps = [#map_weights, #map_v, #map_out],
#       iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
#     } ins(%attention_scores, %value : tensor<1x8x128x128xf32>, tensor<1x8x128x64xf32>) 
#       outs(%output_init : tensor<1x8x128x64xf32>) {
#     ^bb0(%weight: f32, %v: f32, %acc: f32):
#       %weighted = arith.mulf %weight, %v : f32
#       %sum = arith.addf %acc, %weighted : f32
#       linalg.yield %sum : f32
#     } -> tensor<1x8x128x64xf32>
    
#     return %attention_output : tensor<1x8x128x64xf32>
#   }
# }
#         '''
#         return baseline.strip()
    
#     def compile_with_optimizations(self, base_mlir, optimization_params):
#         """Apply real MLIR optimizations and compile"""
#         try:
#             print(f"🔧 Applying optimizations: {optimization_params}")
            
#             # Apply transformation passes with correct syntax
#             optimized_mlir, error = self.compiler.apply_transform_passes(base_mlir, optimization_params)
            
#             if optimized_mlir is None:
#                 return False, f"Optimization failed: {error}"
            
#             print(f"✅ Optimization succeeded, IR size: {len(optimized_mlir)} chars")
#             return True, optimized_mlir
            
#         except Exception as e:
#             return False, f"Optimization error: {e}"
    
#     def get_reference_performance(self):
#         """Get baseline performance using real MLIR compilation"""
#         if self.reference_performance is None:
#             base_mlir = self.create_baseline_mlir()
            
#             # Compile baseline without optimizations
#             baseline_compiled, error = self.compiler.compile_mlir(base_mlir)
#             if baseline_compiled is None:
#                 print(f"❌ Baseline compilation failed: {error}")
#                 # Fallback to estimated performance
#                 self.reference_performance = 5.0
#                 return self.reference_performance
            
#             # Benchmark baseline performance
#             total_time = 0
#             for config in self.test_configs:
#                 runtime, bench_error = self.compiler.benchmark_mlir(baseline_compiled, config)
#                 if bench_error:
#                     print(f"⚠️ Baseline benchmark warning: {bench_error}")
#                 total_time += runtime
            
#             self.reference_performance = total_time / len(self.test_configs)
#             print(f"📊 Reference performance: {self.reference_performance:.4f}")
        
#         return self.reference_performance

# def evaluate_program(program_content):
#     """
#     Main evaluation function using corrected MLIR compilation.
#     Aims to achieve AlphaEvolve's 32% speedup target.
#     """
#     try:
#         # Global evaluator instance using corrected MLIR
#         evaluator = FixedMLIRAttentionEvaluator()

#         # Execute the evolved program to get optimization parameters
#         exec_globals = {}
#         exec(program_content, exec_globals)
        
#         if 'optimize_attention' not in exec_globals:
#             return {"error": 1000.0, "compilation_error": "No optimize_attention function"}
        
#         # Get optimization parameters
#         params = exec_globals['optimize_attention']()
#         print(f"🧬 Evaluating parameters: {params}")
        
#         # Load base MLIR
#         base_mlir = evaluator.create_baseline_mlir()
        
#         # Apply real MLIR optimizations and compile
#         success, optimized_result = evaluator.compile_with_optimizations(base_mlir, params)
        
#         if not success:
#             # Compilation failed - high error penalty
#             print(f"❌ Compilation failed: {optimized_result}")
#             return {"error": 100.0, "compilation_error": str(optimized_result)[:200]}
        
#         # Benchmark optimized performance using real MLIR
#         total_runtime = 0
#         benchmark_errors = []
        
#         for config in evaluator.test_configs:
#             runtime, bench_error = evaluator.compiler.benchmark_mlir(optimized_result, config)
#             if bench_error:
#                 benchmark_errors.append(bench_error)
#             total_runtime += runtime
        
#         avg_runtime = total_runtime / len(evaluator.test_configs)
        
#         # Calculate speedup vs reference
#         reference_time = evaluator.get_reference_performance()
#         speedup = reference_time / avg_runtime if avg_runtime > 0 else 0.0
        
#         # Convert speedup to error metric (targeting AlphaEvolve's 32% improvement)
#         target_speedup = 1.32  # 32% improvement target like AlphaEvolve
        
#         if speedup >= target_speedup:
#             # Achieved target! Error decreases as we exceed target
#             error = max(0.1, (target_speedup - speedup) * 5)
#         else:
#             # Below target - error increases
#             error = (target_speedup - speedup) * 50
        
#         error = max(0.01, error)
        
#         # Prepare detailed result
#         result = {
#             "error": error,
#             "speedup": speedup,
#             "runtime": avg_runtime,
#             "reference_runtime": reference_time,
#             "target_speedup": target_speedup,
#             "achieved_target": speedup >= target_speedup,
#             "real_mlir_compilation": True,
#             "ir_size": len(optimized_result),
#         }
        
#         # Add parameter metrics for analysis
#         for key, value in params.items():
#             if isinstance(value, (int, float, bool)):
#                 result[f"param_{key}"] = float(value) if isinstance(value, bool) else value
        
#         # Add any benchmark warnings
#         if benchmark_errors:
#             result["benchmark_warnings"] = "; ".join(benchmark_errors[:3])
        
#         print(f"📊 Result: error={error:.3f}, speedup={speedup:.3f}x, runtime={avg_runtime:.6f}")
#         if speedup >= target_speedup:
#             print(f"🎯 TARGET ACHIEVED! {speedup:.3f}x >= {target_speedup}x")
#         else:
#             print(f"🎯 Target missed: {speedup:.3f}x < {target_speedup}x")
        
#         return result
        
#     except Exception as e:
#         print(f"❌ Evaluation exception: {e}")
#         return {"error": 1000.0, "exception": str(e)[:200]}

# def evaluate(program_file):
#     """Entry point for evaluation"""
#     try:
#         with open(program_file, 'r') as f:
#             program_content = f.read()
        
#         result = evaluate_program(program_content)
#         print(json.dumps(result, indent=2))
#         return result
#     except Exception as e:
#         error_result = {"error": 1000.0, "exception": str(e)}
#         print(json.dumps(error_result, indent=2))

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python evaluator.py <program_file>")
#         sys.exit(1)
    
# #     evaluate(sys.argv[1])