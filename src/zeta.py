"""RC7 Zeta -- Global Invariant Anchor. SIP License v1.1."""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from fractions import Fraction
from enum import Enum
import time

@dataclass
class EdgeAtom:
    source: int; target: int
    beta: Fraction; kappa: Fraction
    alpha: Fraction; gamma: Fraction; d: Fraction
    @property
    def delta(self) -> Fraction:
        return self.beta * self.kappa - self.alpha * self.gamma
    @property
    def stable(self) -> bool: return self.delta > 0

@dataclass
class SystemState:
    nodes: Set[int]; edges: List[EdgeAtom]
    def adjacency_list(self):
        adj = {n: [] for n in self.nodes}
        for e in self.edges: adj[e.source].append(e.target)
        return adj
    def find_cycles(self, max_len=8):
        adj = self.adjacency_list(); cycles = []
        def dfs(start, cur, path, vis):
            if len(path) > max_len: return
            for nb in adj.get(cur, []):
                if nb == start and len(path) >= 3: cycles.append(list(path))
                elif nb not in vis and nb >= start:
                    vis.add(nb); path.append(nb)
                    dfs(start, nb, path, vis)
                    path.pop(); vis.discard(nb)
        for n in sorted(self.nodes): dfs(n, n, [n], {n})
        return cycles
    def get_edge(self, s, t):
        for e in self.edges:
            if e.source == s and e.target == t: return e
        return None
    def spectral_radius_bound(self):
        adj = self.adjacency_list()
        return Fraction(max((len(v) for v in adj.values()), default=0))

@dataclass
class ZetaResult:
    holds: bool; local_stable: bool; topology_safe: bool
    spectral_contained: bool; cycle_gain_bounded: bool
    delta_min: Fraction; instability_low: bool = True
    instability_risk: float = 0.0
    vulnerable_cycles: List = field(default_factory=list)
    gain_violated_cycles: List = field(default_factory=list)
    spectral_radius: Fraction = Fraction(0)
    spectral_bound: Fraction = Fraction(0)
    details: Dict = field(default_factory=dict)

class Zeta:
    """zeta(S) = Gate1 AND Gate2 AND Gate3 AND Gate4 AND Gate5"""
    def __init__(self, exchange_rate=Fraction(5,4)):
        self.exchange_rate = Fraction(exchange_rate)
    def evaluate(self, state: SystemState) -> ZetaResult:
        # Gate 1: local stability
        dmin = min((e.delta for e in state.edges), default=Fraction(1))
        g1 = all(e.delta > 0 for e in state.edges) if state.edges else True
        # Gate 2: topology safety
        cycles = state.find_cycles(); vuln = []
        for c in cycles:
            if len(c) % 2 == 0:
                for i in range(len(c)):
                    e = state.get_edge(c[i], c[(i+1)%len(c)])
                    if e and e.beta > e.d: vuln.append(c); break
        g2 = len(vuln) == 0
        # Gate 3: spectral containment
        rho = state.spectral_radius_bound()
        dmin_d = min((e.d for e in state.edges), default=Fraction(1))
        rho_star = self.exchange_rate * dmin_d
        g3 = rho < rho_star
        # Gate 4: small-gain on cycles
        bad = []
        for c in cycles:
            gain = Fraction(1)
            for i in range(len(c)):
                e = state.get_edge(c[i], c[(i+1)%len(c)])
                if e:
                    norm_sq = e.beta**2 + e.gamma**2 + e.alpha**2 + e.kappa**2
                    gain *= norm_sq / max(e.d**2, Fraction(1,10**20))
                else: gain = Fraction(0); break
            if gain >= 1: bad.append((c, gain))
        g4 = len(bad) == 0
        g5 = True  # Sigma engine deferred to runtime
        return ZetaResult(holds=g1 and g2 and g3 and g4 and g5,
            local_stable=g1, topology_safe=g2, spectral_contained=g3,
            cycle_gain_bounded=g4, delta_min=dmin,
            vulnerable_cycles=vuln, gain_violated_cycles=bad,
            spectral_radius=rho, spectral_bound=rho_star)

class DeltaType(Enum):
    PARAM_UPDATE="param_update"; ADD_EDGE="add_edge"
    REMOVE_EDGE="remove_edge"; ADD_NODE="add_node"
    REMOVE_NODE="remove_node"

@dataclass
class Delta:
    delta_type: DeltaType; timestamp: float
    edge_source: Optional[int]=None; edge_target: Optional[int]=None
    param_name: Optional[str]=None; old_value=None; new_value=None
    new_edge: Optional[EdgeAtom]=None; removed_edge: Optional[EdgeAtom]=None
    node_id: Optional[int]=None
    def invert(self):
        if self.delta_type == DeltaType.PARAM_UPDATE:
            return Delta(DeltaType.PARAM_UPDATE, time.time(),
                self.edge_source, self.edge_target, self.param_name,
                self.new_value, self.old_value)
        if self.delta_type == DeltaType.ADD_EDGE:
            return Delta(DeltaType.REMOVE_EDGE, time.time(),
                removed_edge=self.new_edge)
        if self.delta_type == DeltaType.REMOVE_EDGE:
            return Delta(DeltaType.ADD_EDGE, time.time(),
                new_edge=self.removed_edge)
        raise ValueError(f"Cannot invert {self.delta_type}")

def apply_delta(state, delta):
    nodes = set(state.nodes)
    edges = [EdgeAtom(e.source,e.target,e.beta,e.kappa,
                      e.alpha,e.gamma,e.d) for e in state.edges]
    if delta.delta_type == DeltaType.PARAM_UPDATE:
        for e in edges:
            if e.source==delta.edge_source and e.target==delta.edge_target:
                setattr(e, delta.param_name, delta.new_value); break
    elif delta.delta_type == DeltaType.ADD_EDGE:
        e = delta.new_edge
        edges.append(EdgeAtom(e.source,e.target,e.beta,e.kappa,
                              e.alpha,e.gamma,e.d))
        nodes.update({e.source, e.target})
    elif delta.delta_type == DeltaType.REMOVE_EDGE:
        e = delta.removed_edge
        edges = [x for x in edges if not(x.source==e.source and x.target==e.target)]
    elif delta.delta_type == DeltaType.ADD_NODE: nodes.add(delta.node_id)
    elif delta.delta_type == DeltaType.REMOVE_NODE:
        nodes.discard(delta.node_id)
        edges = [e for e in edges if e.source!=delta.node_id and e.target!=delta.node_id]
    return SystemState(nodes=nodes, edges=edges)

class ZetaGuard:
    def __init__(self, exchange_rate=Fraction(5,4)):
        self.zeta = Zeta(exchange_rate)
    def validate(self, state, delta):
        z_before = self.zeta.evaluate(state)
        z_after = self.zeta.evaluate(apply_delta(state, delta))
        violations = []
        if not z_after.local_stable: violations.append("Gate 1: local_stable")
        if not z_after.topology_safe: violations.append("Gate 2: topology_safe")
        if not z_after.spectral_contained: violations.append("Gate 3: spectral")
        if not z_after.cycle_gain_bounded: violations.append("Gate 4: gain")
        return {"valid": z_after.holds, "violations": violations,
                "before": z_before, "after": z_after}
