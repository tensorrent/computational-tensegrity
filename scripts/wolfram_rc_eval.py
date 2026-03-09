#!/usr/bin/env python3
"""
Wolfram Hypergraph RC Stack Evaluation — v2 (Fixed)
====================================================
Fixes from v1:
  - Symmetrise adjacency matrix (directed hyperedges → undirected graph)
  - Use Laplacian eigenvalues directly for stability
  - Extended step counts for meaningful spectral evolution
  - Causal invariance test: run same rule with different update orderings
"""

import numpy as np
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Optional
import itertools
import time
import json

# ═══════════════════════════════════════════════════════════════
# WOLFRAM ENGINE (same as v1, compact)
# ═══════════════════════════════════════════════════════════════

class HyperGraph:
    def __init__(self, edge_tuples):
        self._edges = list(edge_tuples)
    
    @property
    def nodes(self):
        s = set()
        for e in self._edges:
            s.update(e)
        return s
    
    @property
    def n_nodes(self): return len(self.nodes)
    
    @property
    def n_edges(self): return len(self._edges)
    
    def adjacency_symmetric(self):
        """Symmetrised adjacency: for edge (a,b), add A[a,b] and A[b,a]."""
        nodes = sorted(self.nodes)
        n = len(nodes)
        if n == 0: return np.zeros((0,0)), nodes
        idx = {v: i for i, v in enumerate(nodes)}
        A = np.zeros((n, n))
        for e in self._edges:
            for i in range(len(e)-1):
                a, b = e[i], e[i+1]
                if a in idx and b in idx:
                    A[idx[a], idx[b]] += 1
                    A[idx[b], idx[a]] += 1  # symmetrise
        return A, nodes
    
    def laplacian(self):
        A, nodes = self.adjacency_symmetric()
        D = np.diag(A.sum(axis=1))
        return D - A, nodes
    
    def copy(self):
        return HyperGraph(list(self._edges))


class WolframEngine:
    def __init__(self, pattern, replacement, initial):
        self.pattern = pattern
        self.replacement = replacement
        self.state = initial.copy()
        self.history = [initial.copy()]
        self._next = max(initial.nodes) + 1 if initial.nodes else 0
    
    def _find_matches(self):
        edges = self.state._edges
        matches = []
        if len(self.pattern) > len(edges): return []
        
        for combo in itertools.permutations(range(len(edges)), len(self.pattern)):
            binding = {}
            valid = True
            for pi, ei in enumerate(combo):
                pe, ge = self.pattern[pi], edges[ei]
                if len(pe) != len(ge):
                    valid = False; break
                for pv, gv in zip(pe, ge):
                    if pv in binding:
                        if binding[pv] != gv:
                            valid = False; break
                    else:
                        binding[pv] = gv
                if not valid: break
            if valid and binding not in matches:
                matches.append(binding)
        return matches
    
    def step(self, strategy="first"):
        matches = self._find_matches()
        if not matches: return False
        
        if strategy == "first":
            binding = matches[0]
        elif strategy == "random":
            binding = matches[np.random.randint(len(matches))]
        elif strategy == "last":
            binding = matches[-1]
        else:
            binding = matches[0]
        
        # Remove matched edges
        remove = []
        for pe in self.pattern:
            remove.append(tuple(binding[v] for v in pe))
        
        # Create replacement edges with new nodes for unbound vars
        new_binding = dict(binding)
        for re in self.replacement:
            for v in re:
                if v not in new_binding:
                    new_binding[v] = self._next
                    self._next += 1
        
        new_edges = [tuple(new_binding[v] for v in re) for re in self.replacement]
        remaining = [e for e in self.state._edges if tuple(e) not in remove]
        self.state = HyperGraph(remaining + new_edges)
        self.history.append(self.state.copy())
        return True
    
    def evolve(self, steps, strategy="first"):
        for _ in range(steps):
            if not self.step(strategy): break
        return self.history


# ═══════════════════════════════════════════════════════════════
# RC STACK EVALUATION (v2 — fixed)
# ═══════════════════════════════════════════════════════════════

def evaluate(graph, step=0):
    """Full RC stack evaluation on a hypergraph state."""
    A, nodes = graph.adjacency_symmetric()
    n = len(nodes)
    
    if n < 2:
        return {"step": step, "n": n, "edges": graph.n_edges,
                "rho": 0, "gap": 0, "max_deg": 0, "mean_deg": 0,
                "rc4": True, "rc6": True, "rc7": True, "sigma": 0,
                "zeta": True, "eigs": []}
    
    L, _ = graph.laplacian()
    
    try:
        adj_eigs = np.sort(np.real(np.linalg.eigvals(A)))[::-1]
        lap_eigs = np.sort(np.real(np.linalg.eigvals(L)))
    except:
        adj_eigs = np.zeros(n)
        lap_eigs = np.zeros(n)
    
    rho = float(np.max(np.abs(adj_eigs)))
    gap = float(lap_eigs[1]) if n > 1 and len(lap_eigs) > 1 else 0
    
    degrees = A.sum(axis=1)
    max_deg = int(np.max(degrees))
    mean_deg = float(np.mean(degrees))
    
    # RC4: local stability — spectral radius bounded by mean degree
    rc4 = rho < 2 * mean_deg if mean_deg > 0 else True
    
    # RC6: spectral containment — gap should be positive (connected)
    rc6 = gap > 0.05
    
    # RC7: gain bounded — spectral radius growth rate bounded
    rc7 = rho < n * 0.5  # conservative bound
    
    # Sigma: perturbation probe
    sigma = 0.0
    if n > 2 and n < 200:
        try:
            J = A / max(rho, 1.0) * 0.95
            escapes = 0
            for _ in range(30):
                x = np.zeros(n)
                for _ in range(100):
                    x = J @ x + 0.05 * np.random.randn(n)
                    if np.linalg.norm(x) > 10:
                        escapes += 1; break
            sigma = escapes / 30
        except:
            pass
    
    zeta = rc4 and rc6 and rc7 and (sigma < 0.5)
    
    return {"step": step, "n": n, "edges": graph.n_edges,
            "rho": rho, "gap": gap, "max_deg": max_deg, 
            "mean_deg": mean_deg,
            "rc4": rc4, "rc6": rc6, "rc7": rc7, "sigma": sigma,
            "zeta": zeta, "eigs": adj_eigs[:5].tolist()}


# ═══════════════════════════════════════════════════════════════
# CAUSAL INVARIANCE TEST
# ═══════════════════════════════════════════════════════════════

def test_causal_invariance(pattern, replacement, initial, steps, n_orderings=5):
    """
    Test causal invariance: do different update orderings produce
    the same spectral properties?
    
    Wolfram claims causal invariance → same causal graph regardless 
    of update order. We test the weaker condition: same spectral 
    properties (eigenvalue distribution) regardless of update order.
    """
    results = []
    
    strategies = ["first", "random", "random", "random", "last"][:n_orderings]
    
    for i, strategy in enumerate(strategies):
        np.random.seed(42 + i)  # different but reproducible
        engine = WolframEngine(pattern, replacement, initial)
        engine.evolve(steps, strategy=strategy)
        
        final = engine.history[-1]
        ev = evaluate(final, step=steps)
        results.append({
            "strategy": strategy,
            "seed": 42 + i,
            "n_nodes": ev["n"],
            "n_edges": ev["edges"],
            "rho": ev["rho"],
            "gap": ev["gap"],
            "zeta": ev["zeta"],
        })
    
    # Check if spectral properties are invariant across orderings
    rhos = [r["rho"] for r in results]
    gaps = [r["gap"] for r in results]
    node_counts = [r["n_nodes"] for r in results]
    
    rho_spread = max(rhos) - min(rhos) if rhos else 0
    gap_spread = max(gaps) - min(gaps) if gaps else 0
    node_spread = max(node_counts) - min(node_counts) if node_counts else 0
    
    # Causal invariance holds if spectral properties are identical
    # (up to numerical tolerance)
    invariant = rho_spread < 0.01 and gap_spread < 0.01 and node_spread == 0
    
    return {
        "orderings": results,
        "rho_spread": rho_spread,
        "gap_spread": gap_spread,
        "node_spread": node_spread,
        "causal_invariant": invariant,
    }


# ═══════════════════════════════════════════════════════════════
# RUN ALL EVALUATIONS
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RC STACK EVALUATION OF WOLFRAM HYPERGRAPH MODELS — v2 (Fixed)")
    print("=" * 70)
    
    rules = {
        "Rule 1: Self-reproducing": {
            "pattern": [(0,1), (0,2)],
            "replacement": [(0,1), (0,3), (1,3), (2,3)],
            "initial": HyperGraph([(0,1), (0,2), (0,3)]),
            "steps": 20,
        },
        "Rule 2: Simple growth": {
            "pattern": [(0,1)],
            "replacement": [(0,1), (1,2)],
            "initial": HyperGraph([(0,1)]),
            "steps": 25,
        },
        "Rule 3: Universe geometry": {
            "pattern": [(0,1), (1,2)],
            "replacement": [(3,1), (1,2), (2,3), (0,3)],
            "initial": HyperGraph([(0,1), (1,2), (2,3), (3,0)]),
            "steps": 18,
        },
    }
    
    all_data = {}
    
    for label, cfg in rules.items():
        print(f"\n{'━'*70}")
        print(f"  {label}")
        print(f"{'━'*70}")
        
        engine = WolframEngine(cfg["pattern"], cfg["replacement"], cfg["initial"])
        t0 = time.time()
        history = engine.evolve(cfg["steps"])
        t1 = time.time()
        
        print(f"  Evolution: {len(history)} states in {(t1-t0)*1000:.0f}ms")
        print(f"  Final: {history[-1].n_nodes} nodes, {history[-1].n_edges} edges")
        
        evals = []
        print(f"\n  {'Step':>4} {'N':>5} {'E':>5} {'ρ(A)':>7} {'λ₂':>7} "
              f"{'deg':>5} {'RC4':>4} {'RC6':>4} {'RC7':>4} {'Σ':>5} {'ζ':>3}")
        print(f"  {'─'*4} {'─'*5} {'─'*5} {'─'*7} {'─'*7} "
              f"{'─'*5} {'─'*4} {'─'*4} {'─'*4} {'─'*5} {'─'*3}")
        
        for i, state in enumerate(history):
            ev = evaluate(state, i)
            evals.append(ev)
            
            if i % max(1, len(history)//12) == 0 or i == len(history)-1:
                s = lambda b: "✓" if b else "✗"
                print(f"  {i:>4} {ev['n']:>5} {ev['edges']:>5} "
                      f"{ev['rho']:>7.3f} {ev['gap']:>7.3f} "
                      f"{ev['mean_deg']:>5.1f} {s(ev['rc4']):>4} "
                      f"{s(ev['rc6']):>4} {s(ev['rc7']):>4} "
                      f"{ev['sigma']:>5.2f} {s(ev['zeta']):>3}")
        
        # Trends
        rhos = [e["rho"] for e in evals]
        gaps = [e["gap"] for e in evals if e["gap"] > 0]
        zeta_pass = sum(1 for e in evals if e["zeta"])
        
        print(f"\n  Analysis:")
        print(f"    ρ(A): {rhos[0]:.3f} → {rhos[-1]:.3f}")
        if gaps:
            print(f"    λ₂:   {gaps[0]:.3f} → {gaps[-1]:.3f}")
        print(f"    ζ passed: {zeta_pass}/{len(evals)}")
        
        if len(rhos) > 3:
            slope = np.polyfit(range(len(rhos)), rhos, 1)[0]
            print(f"    ρ trend: {'GROWING' if slope > 0.05 else 'STABLE' if abs(slope) < 0.05 else 'DECAYING'} (slope={slope:.4f})")
        
        # Tensegrity ratio: ρ/λ₂
        if gaps:
            final_ratio = rhos[-1] / max(gaps[-1], 0.001)
            print(f"    Tensegrity ratio ρ/λ₂: {final_ratio:.2f} "
                  f"({'FRAGILE' if final_ratio > 10 else 'HEALTHY'})")
        
        all_data[label] = evals
        
        # ── Causal invariance test ──
        print(f"\n  Causal Invariance Test (5 orderings, {min(cfg['steps'], 10)} steps):")
        ci = test_causal_invariance(
            cfg["pattern"], cfg["replacement"], cfg["initial"],
            min(cfg["steps"], 10), n_orderings=5
        )
        
        print(f"    ρ spread:    {ci['rho_spread']:.4f}")
        print(f"    λ₂ spread:   {ci['gap_spread']:.4f}")
        print(f"    Node spread: {ci['node_spread']}")
        print(f"    Causal invariant: {'YES ✓' if ci['causal_invariant'] else 'NO ✗'}")
        
        if not ci["causal_invariant"]:
            print(f"    Per-ordering results:")
            for r in ci["orderings"]:
                print(f"      {r['strategy']:>8} seed={r['seed']}: "
                      f"N={r['n_nodes']}, E={r['n_edges']}, "
                      f"ρ={r['rho']:.3f}, λ₂={r['gap']:.3f}")
    
    # ── Summary ──
    print(f"\n{'═'*70}")
    print("SUMMARY")
    print(f"{'═'*70}")
    
    print(f"\n  {'Rule':<30} {'Final ρ':>8} {'Final λ₂':>9} {'ζ%':>6} {'Causal?':>8}")
    print(f"  {'─'*30} {'─'*8} {'─'*9} {'─'*6} {'─'*8}")
    
    for label, evals in all_data.items():
        final = evals[-1]
        zp = 100 * sum(1 for e in evals if e["zeta"]) / len(evals)
        ci = test_causal_invariance(
            rules[label]["pattern"], rules[label]["replacement"],
            rules[label]["initial"], min(rules[label]["steps"], 10), 3
        )
        print(f"  {label:<30} {final['rho']:>8.3f} {final['gap']:>9.3f} "
              f"{zp:>5.1f}% {'YES' if ci['causal_invariant'] else 'NO':>8}")
    
    # Save data
    save_data = {}
    for label, evals in all_data.items():
        save_data[label] = [{k: v for k, v in e.items() if k != "eigs"} for e in evals]
    
    with open("scripts/wolfram_data.json", "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nData saved to wolfram_data.json")
    return all_data


if __name__ == "__main__":
    main()
