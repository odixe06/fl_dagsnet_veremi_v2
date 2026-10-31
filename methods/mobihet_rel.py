"""MobiHet with class/client-batch reliability gating on teacher-to-private KD only."""
from methods.mobihet import Method as MobiHet


class Method(MobiHet):
    name = "mobihet_rel"
    reliability_gate = True
