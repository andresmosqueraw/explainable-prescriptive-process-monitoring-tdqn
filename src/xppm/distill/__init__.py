"""Policy distillation: TDQN -> Decision Tree surrogate."""

from xppm.distill.distill_policy import distill_policy
from xppm.distill.export_rules import export_rules

__all__ = ["distill_policy", "export_rules"]
