"""RC Stack Turn-Key Demo. SIP License v1.1."""
from fractions import Fraction
# from rc_stack.zeta import *  (imports shown inline for paper)

def demo():
    guard = ZetaGuard(exchange_rate=Fraction(5,4))
    state = SystemState(nodes={0,1,2}, edges=[
        EdgeAtom(0,1, Fraction(3,4),Fraction(3,4),Fraction(1,5),Fraction(1,5),Fraction(1)),
        EdgeAtom(1,2, Fraction(1,2),Fraction(1,2),Fraction(1,4),Fraction(1,4),Fraction(1))])
    z = guard.zeta.evaluate(state)
    print(f"zeta(S) = {z.holds}")
    # Safe mutation: reduce cross-coupling
    safe = Delta(DeltaType.PARAM_UPDATE, 0.0, 0, 1, "alpha",
                 Fraction(1,5), Fraction(1,10))
    r = guard.validate(state, safe)
    print(f"Safe delta: valid={r['valid']}")
    # Hazardous mutation: add high-gain cycle edge
    hazard_edge = EdgeAtom(2,0, Fraction(3),Fraction(3),
                           Fraction(1,10),Fraction(1,10),Fraction(1))
    bad = Delta(DeltaType.ADD_EDGE, 1.0, new_edge=hazard_edge)
    r = guard.validate(state, bad)
    print(f"Hazard delta: valid={r['valid']}, violations={r['violations']}")
