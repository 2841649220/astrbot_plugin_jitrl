"""JitRL continual-learning core for the AstrBot agent.

The package mirrors Algorithm 1 of *Just-In-Time Reinforcement Learning*
(Li et al., ICML 2026). Every module maps onto one stage of that algorithm:

* `state`     -> `AbstractState(o)` (state construction, Appendix F)
* `memory`    -> the non-parametric memory M = {(s_i, a_i, G_i)}
* `estimator` -> steps 1-5: retrieval, V/Q estimation, advantage normalisation
* `policy`    -> step 6: the closed-form logit update z' = z + beta * advantage
* `evaluator` -> step 7: the LLM evaluator E: tau -> {r_t} and discounted returns
* `trajectory`-> episode reconstruction from an AstrBot agent run
* `config`    -> typed, default-filled view over `_conf_schema.json`
"""

from __future__ import annotations

__version__ = "0.1.4"

__all__ = ["__version__"]
