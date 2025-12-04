"""
REMOVED: duplicate smoke-evaluation helper

The evaluation logic has been consolidated into `src/evaluate.py`.
If you previously used `src/evaluate_smoke.py` please switch to:

  from src.evaluate import evaluate_smoke
  evaluate_smoke()

This file remains as a stub to avoid accidental execution of old code.
"""

raise RuntimeError('src/evaluate_smoke.py was removed; use src/evaluate.evaluate_smoke instead')
