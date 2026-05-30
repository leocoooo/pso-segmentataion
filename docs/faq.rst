===
FAQ
===

**Q: Which API should I use - OO or functional?**

A: Use the functional API (`segment_scores`) for quick experiments and the OO API (`SegmentationOptimizer`) for production code where you need configuration control and state management.

---

**Q: How do I choose the number of segments?**

A: Start with 4-5 segments and adjust based on business requirements. More segments provide finer granularity but may overfit. Validate results on holdout data.

---

**Q: What's a good PSO population size?**

A: Start with 50-100. Larger populations explore better but are slower. For 5 segments, 50-100 is typical. For 3 segments, 30-50 may suffice.

---

**Q: How many PSO iterations do I need?**

A: 200-500 iterations usually suffice for convergence. Monitor the fitness history - if still improving at max_iter, increase it.

---

**Q: Can I enforce monotonic target mean?**

A: Yes, via the objective function. Use `make_objective(..., penalties=[monotonic_penalty(weight=...)])` or provide your own penalty callable. The built-in penalty checks the segment-level target mean.

---

**Q: How do I handle imbalanced data?**

A: If your target is binary, the segment target means reflect class imbalance. For extremely imbalanced data, add a balance penalty to the fitness function.

---

**Q: Can I use custom fitness functions?**

A: Yes. The recommended path is `make_objective(scores, labels, metric=..., penalties=...)`. You can also pass any callable with signature `fitness(cuts) -> float`. See :doc:`advanced` for patterns and examples.

---

**Q: How do I export and load results?**

A: Use `export_segmentation_to_csv()` and `import_segmentation_from_csv()` for CSV, or `save_optimizer_state()` and `load_optimizer_state()` for pickle.

---

**Q: What's the difference between R², H_inter, and H_intra?**

A: R² measures explained variance. H_inter measures segment separation (between-group). H_intra measures segment homogeneity (within-group).

---

**Q: Can I use numpy arrays with different dtypes?**

A: Yes, but they're automatically converted to float64. Ensure your data is numeric (0-1 for binary labels, 0-1 for scores).

---

**Q: How do I reproduce results?**

A: Set the `seed` parameter in `OptimizerConfig`. Same seed produces identical results.

---

**Q: Is pso-segmentation thread-safe?**

A: The PSO algorithm itself is deterministic with a fixed seed. However, without a seed, results vary due to randomization. Use locks if sharing state across threads.

---

**Q: Can I use this with large datasets (>1M samples)?**

A: Yes, but PSO evaluation scales with dataset size. For very large datasets, consider stratified sampling for fitness evaluation or pre-binning scores.

---

**Q: What Python versions are supported?**

A: Python 3.12+. The package uses PEP 604 type hints (`X | None`) and PEP 563 (`from __future__ import annotations`).

---

**Q: Can I run multiple segmentations in parallel?**

A: Yes, create separate `SegmentationOptimizer` instances and run them concurrently. Each instance is stateless between fit() calls.

---

**Q: How do I tune PSO parameters?**

A: See :doc:`advanced` for parameter effects. Start with defaults and adjust based on convergence behavior and speed requirements.

---

**Q: Can I warm-start PSO?**

A: Currently no built-in support, but you could extend PSO to accept initial particle positions. File an issue if this is needed.

---

**Q: Is there support for categorical features?**

A: Not directly. Pre-process by computing risk scores from categorical features (e.g., WOE binning, logistic regression), then segment scores.

---

**Q: Can I segment on multiple dimensions?**

A: The current design segments on a 1D score. For multi-dimensional segmentation, compute a composite score or extend the fitness function logic.

---

**Q: How do I handle missing values?**

A: Remove rows with missing scores or labels before segmentation. The PSO algorithm requires complete data.

---

**Q: Can I use different distance metrics in H_inter/H_intra?**

A: Not without custom implementation. The built-in metrics use specific definitions suitable for credit scoring. Extend the code if needed.

---

**Q: What's the computational complexity?**

A: O(pop_size × max_iter × n_samples) for PSO evaluation. Typical runtime: 10-100ms for n_samples=10K, pop_size=50, max_iter=200.

---

**Q: Do you have examples for [my use case]?**

A: Check :doc:`examples` and :doc:`advanced`. If not covered, consider filing an issue or PR to add it!

---

See :doc:`getting_started` for installation and quick start, and :doc:`api/index` for complete API documentation.
