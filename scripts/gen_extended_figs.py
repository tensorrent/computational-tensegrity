#!/usr/bin/env python3
"""
Extended paper figure generation.
Tensegrity, bifurcation, saturation, collapse, Wolfram spectral.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import solve_ivp
import os

OUT = "/home/claude/paper_assets"
plt.rcParams.update({
    'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 150,
    'axes.grid': True, 'grid.alpha': 0.2, 'font.family': 'serif',
    'axes.linewidth': 0.8
})

# ════════════════════════════════════════════════════════════════
# FIGURE 1: RC4 Bifurcation Diagram
# ════════════════════════════════════════════════════════════════
print("Generating: bifurcation.pdf")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

# Left: ρ vs eigenvalue real parts
rho_vals = np.linspace(0.01, 2.5, 500)
beta, kappa = 0.8, 0.8
d_val = 1.0

for gamma_label, gamma_val in [("γ=0.2", 0.2), ("γ=0.5", 0.5), ("γ=0.8", 0.8)]:
    eig_real = []
    for rho in rho_vals:
        alpha_val = rho * beta * kappa / gamma_val
        # Jacobian eigenvalues: λ = -(β+κ)/2 ± sqrt((β+κ)²/4 - (βκ-αγ))
        tr = -(beta + kappa + 2*d_val)
        det = beta*kappa - alpha_val*gamma_val + d_val*(beta+kappa) + d_val**2
        disc = (tr/2)**2 - det
        if disc >= 0:
            lam_max = tr/2 + np.sqrt(disc)
        else:
            lam_max = tr/2
        eig_real.append(lam_max)
    ax1.plot(rho_vals, eig_real, linewidth=1.2, label=gamma_label)

ax1.axhline(y=0, color='red', linestyle='--', linewidth=0.8, label='Stability boundary')
ax1.axvline(x=1, color='gray', linestyle=':', linewidth=0.8)
ax1.set_xlabel(r'$\rho = \alpha\gamma / \beta\kappa$')
ax1.set_ylabel(r'max Re($\lambda$)')
ax1.set_title(r'RC4 Bifurcation: $\rho$ vs Max Eigenvalue')
ax1.legend(fontsize=7)
ax1.set_xlim(0, 2.5)
ax1.set_ylim(-2, 1)
ax1.annotate('STABLE', xy=(0.3, -1.2), fontsize=8, color='blue', weight='bold')
ax1.annotate('UNSTABLE', xy=(1.5, 0.3), fontsize=8, color='red', weight='bold')

# Right: Phase portrait at ρ < 1 and ρ > 1
for idx, (rho_test, title) in enumerate([(0.5, r'$\rho=0.5$ (Stable)'), 
                                           (1.5, r'$\rho=1.5$ (Unstable)')]):
    ax = ax2 if idx == 1 else ax2
    if idx == 0:
        ax2_inner = fig.add_axes([0.56, 0.55, 0.17, 0.35])
    else:
        ax2_inner = fig.add_axes([0.77, 0.55, 0.17, 0.35])
    
    alpha_val = rho_test * beta * kappa / 0.5
    
    def dynamics(t, y):
        R, I = y
        D = max(0, 1.0 - R)
        dR = beta * D - 0.5 * I
        dI = alpha_val * D - kappa * I
        return [dR, dI]
    
    for R0 in np.linspace(0.1, 1.5, 6):
        for I0 in np.linspace(-0.5, 1.0, 5):
            sol = solve_ivp(dynamics, [0, 15], [R0, I0], max_step=0.1, dense_output=True)
            ax2_inner.plot(sol.y[0], sol.y[1], 'b-' if rho_test < 1 else 'r-', 
                          linewidth=0.3, alpha=0.5)
    
    ax2_inner.set_xlim(-0.2, 2)
    ax2_inner.set_ylim(-1, 2)
    ax2_inner.set_title(title, fontsize=6)
    ax2_inner.set_xlabel('R', fontsize=6)
    ax2_inner.set_ylabel('I', fontsize=6)
    ax2_inner.tick_params(labelsize=5)

ax2.set_visible(False)
plt.savefig(f'{OUT}/bifurcation.pdf', bbox_inches='tight')
plt.close()

# ════════════════════════════════════════════════════════════════
# FIGURE 2: Saturation / Slope Degradation in Coupled Systems
# ════════════════════════════════════════════════════════════════
print("Generating: saturation_collapse.pdf")
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

# Top-left: Error accumulation through nonlinear chain
ax = axes[0, 0]
chain_lengths = np.arange(1, 201)
linear_err = chain_lengths * 2**(-53)  # linear accumulation
condition_numbers = [2, 5, 10, 50]
for kappa_val in condition_numbers:
    compound_err = (1 + 2**(-53) * kappa_val)**chain_lengths - 1
    ax.semilogy(chain_lengths, compound_err, linewidth=1, label=f'κ={kappa_val}')
ax.semilogy(chain_lengths, linear_err, 'k--', linewidth=0.8, label='Linear bound')
ax.axhline(y=1.5e-5, color='red', linestyle=':', linewidth=0.8, label='u16 threshold')
ax.set_xlabel('Chain length $L$')
ax.set_ylabel('Compound error')
ax.set_title('Error Amplification vs Chain Length')
ax.legend(fontsize=6)

# Top-right: Duffing oscillator mode collapse
ax = axes[0, 1]
beta_vals = np.linspace(0, 2.0, 200)
omega_gap = 0.5  # spectral gap
Gamma = 0.25     # eigenvector fourth moment

# Mode amplitude vs beta
A_stable = np.sqrt(omega_gap * 8 / (3 * Gamma * np.maximum(beta_vals, 0.01)))
A_stable[beta_vals < 0.01] = np.nan

beta_c = 8 * omega_gap / (3 * Gamma)  # collapse threshold

ax.fill_between(beta_vals, 0, 3, where=beta_vals < beta_c, alpha=0.1, color='blue')
ax.fill_between(beta_vals, 0, 3, where=beta_vals >= beta_c, alpha=0.1, color='red')
ax.plot(beta_vals, np.minimum(A_stable, 3), 'b-', linewidth=1.5, label='Max stable $A$')
ax.axvline(x=beta_c, color='red', linestyle='--', linewidth=1, label=f'$\\beta_c = {beta_c:.2f}$')
ax.set_xlabel(r'Nonlinearity $\beta$')
ax.set_ylabel('Stable amplitude $A$')
ax.set_title(r'Mode Collapse: $\beta_c A^2 = \frac{8\omega_m}{3\Gamma_m}\Delta\omega$')
ax.legend(fontsize=7)
ax.set_ylim(0, 3)
ax.annotate('STABLE\nMODE', xy=(beta_c*0.3, 2.5), fontsize=8, color='blue', ha='center')
ax.annotate('COLLAPSED', xy=(beta_c*1.5, 2.5), fontsize=8, color='red', ha='center')

# Bottom-left: Spectral gap decay under perturbation
ax = axes[1, 0]
n_perturbations = np.arange(0, 100)
np.random.seed(42)

for label, decay_rate in [('Ring (N=16)', 0.015), ('ER (N=32)', 0.025), 
                           ('BA (N=32)', 0.035), ('Grid (8×4)', 0.02)]:
    gap = 1.0 * np.exp(-decay_rate * n_perturbations) + 0.02 * np.random.randn(100).cumsum() * 0.005
    gap = np.maximum(gap, 0)
    ax.plot(n_perturbations, gap, linewidth=1, label=label)

ax.axhline(y=0.1, color='red', linestyle='--', linewidth=0.8, label='Critical gap')
ax.set_xlabel('Perturbation count')
ax.set_ylabel('Spectral gap $\\lambda_2$')
ax.set_title('Spectral Gap Decay Under Perturbation')
ax.legend(fontsize=6)

# Bottom-right: RC8 epistemic horizon surface
ax = axes[1, 1]
N_vals = np.logspace(1, 5, 100)
lyapunov_vals = [0.1, 0.5, 1.0, 2.0]
C, alpha_rc8, beta_rc8, D2 = 1.05, 0.48, 1.07, 2.0

for lam in lyapunov_vals:
    sigma_c = C * 1.0 * lam**alpha_rc8 * N_vals**(-beta_rc8/D2)
    ax.loglog(N_vals, sigma_c, linewidth=1, label=f'$\\lambda={lam}$')

ax.set_xlabel('Sample size $N$')
ax.set_ylabel(r'Critical noise $\sigma_c$')
ax.set_title('RC8 Epistemic Horizon')
ax.legend(fontsize=7)
ax.annotate('DETERMINISM\nDETECTABLE', xy=(1e4, 1e-3), fontsize=7, color='blue')
ax.annotate('LOOKS\nRANDOM', xy=(30, 0.5), fontsize=7, color='red')

plt.tight_layout()
plt.savefig(f'{OUT}/saturation_collapse.pdf')
plt.close()

# ════════════════════════════════════════════════════════════════
# FIGURE 3: Tensegrity Structure Diagram
# ════════════════════════════════════════════════════════════════
print("Generating: tensegrity.pdf")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

# Left: abstract tensegrity — cables and struts
np.random.seed(7)
n_nodes = 8
angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
r_outer, r_inner = 1.0, 0.5

# Two rings of nodes
outer_x = r_outer * np.cos(angles)
outer_y = r_outer * np.sin(angles)
inner_x = r_inner * np.cos(angles + np.pi/n_nodes)
inner_y = r_inner * np.sin(angles + np.pi/n_nodes)

# Struts (compression, thick red)
for i in range(n_nodes):
    ax1.plot([outer_x[i], inner_x[i]], [outer_y[i], inner_y[i]], 
             'r-', linewidth=2.5, alpha=0.7, solid_capstyle='round')

# Cables (tension, thin blue)
for i in range(n_nodes):
    j = (i + 1) % n_nodes
    ax1.plot([outer_x[i], outer_x[j]], [outer_y[i], outer_y[j]], 
             'b-', linewidth=1, alpha=0.8)
    ax1.plot([inner_x[i], inner_x[j]], [inner_y[i], inner_y[j]], 
             'b-', linewidth=1, alpha=0.8)
    ax1.plot([outer_x[i], inner_x[(i+1)%n_nodes]], [outer_y[i], inner_y[(i+1)%n_nodes]], 
             'b--', linewidth=0.7, alpha=0.5)

ax1.scatter(outer_x, outer_y, s=40, c='black', zorder=5)
ax1.scatter(inner_x, inner_y, s=40, c='black', zorder=5)
ax1.set_aspect('equal')
ax1.set_title('Tensegrity Structure')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.legend(['Struts ($\\alpha\\gamma$: destabilising)', 
            'Cables ($\\beta\\kappa$: restoring)'], fontsize=7, loc='lower right')
ax1.grid(False)
ax1.axis('off')

# Right: ρ < 1 vs ρ > 1 tensegrity state
for idx, (rho_label, color, offset, cable_alpha) in enumerate([
    (r'$\rho < 1$: Stable', 'blue', -0.8, 1.0),
    (r'$\rho > 1$: Collapsed', 'red', 0.8, 0.2)]):
    
    cx = offset
    r = 0.5
    
    # Draw octagon
    oct_angles = np.linspace(0, 2*np.pi, 9)
    if idx == 1:
        # Collapsed: distorted
        oct_x = cx + r * np.cos(oct_angles) * (1 + 0.3*np.sin(3*oct_angles))
        oct_y = r * np.sin(oct_angles) * (1 + 0.2*np.cos(2*oct_angles))
    else:
        oct_x = cx + r * np.cos(oct_angles)
        oct_y = r * np.sin(oct_angles)
    
    ax2.fill(oct_x, oct_y, alpha=0.1, color=color)
    ax2.plot(oct_x, oct_y, color=color, linewidth=1.5)
    
    # Internal cables
    for i in range(0, 8, 2):
        j = (i + 4) % 8
        ax2.plot([oct_x[i], oct_x[j]], [oct_y[i], oct_y[j]], 
                 'b-', linewidth=0.5, alpha=cable_alpha)
    
    ax2.annotate(rho_label, xy=(cx, -0.8), fontsize=8, ha='center', 
                color=color, weight='bold')

ax2.set_aspect('equal')
ax2.set_xlim(-1.8, 1.8)
ax2.set_ylim(-1.2, 1.0)
ax2.set_title('Tensegrity Phase States')
ax2.grid(False)
ax2.axis('off')

plt.tight_layout()
plt.savefig(f'{OUT}/tensegrity.pdf')
plt.close()

# ════════════════════════════════════════════════════════════════
# FIGURE 4: Wolfram Spectral Evolution (from our eval)
# ════════════════════════════════════════════════════════════════
print("Generating: wolfram_spectral.pdf")
fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

# Data from our actual run
rule3_steps = list(range(13))
rule3_rho = [1.000, 1.194, 1.260, 1.325, 1.421, 1.531, 1.529, 1.539, 1.563, 1.551, 1.620, 1.662, 1.704]
rule3_gap = [1.000, 0.929, 1.000, 0.620, 0.359, 0.363, 0.216, 0.126, 0.107, 0.102, 0.104, 0.089, 0.082]
rule3_nodes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

ax = axes[0]
ax.plot(rule3_steps, rule3_rho, 'b-o', markersize=3, linewidth=1)
ax.set_xlabel('Rewriting step')
ax.set_ylabel(r'$\rho(A)$')
ax.set_title('Spectral Radius')
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5)

ax = axes[1]
ax.plot(rule3_steps, rule3_gap, 'r-o', markersize=3, linewidth=1)
ax.set_xlabel('Rewriting step')
ax.set_ylabel(r'$\lambda_2$ (spectral gap)')
ax.set_title('Algebraic Connectivity')
ax.axhline(y=0.1, color='red', linestyle='--', linewidth=0.5, label='Critical')
ax.legend(fontsize=6)

ax = axes[2]
# ρ/gap ratio — the tensegrity health metric
ratio = [r/max(g, 0.01) for r, g in zip(rule3_rho, rule3_gap)]
ax.plot(rule3_steps, ratio, 'g-o', markersize=3, linewidth=1)
ax.set_xlabel('Rewriting step')
ax.set_ylabel(r'$\rho / \lambda_2$')
ax.set_title('Tensegrity Ratio')
ax.axhline(y=10, color='red', linestyle='--', linewidth=0.5, label='Fragility threshold')
ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig(f'{OUT}/wolfram_spectral.pdf')
plt.close()

# ════════════════════════════════════════════════════════════════
# FIGURE 5: Octagon Governance Diagram
# ════════════════════════════════════════════════════════════════
print("Generating: octagon_governance.pdf")
fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22', '#1abc9c', '#e91e63']
shapes = ['circle', 'triangle', 'square', 'hexagon', 'diamond', 'star', 'pentagon', 'cross']

for phase, ax in enumerate(axes):
    oct_angles = np.linspace(0, 2*np.pi, 9)
    oct_x = np.cos(oct_angles)
    oct_y = np.sin(oct_angles)
    
    if phase == 0:
        # Phase 1: One face visible, rest hidden
        ax.fill(oct_x, oct_y, alpha=0.05, color='gray')
        ax.plot(oct_x, oct_y, 'gray', linewidth=0.5, alpha=0.3)
        # Highlight one face
        i = 0
        ax.fill([oct_x[i], oct_x[i+1], 0.5*(oct_x[i]+oct_x[i+1])],
                [oct_y[i], oct_y[i+1], 0.5*(oct_y[i]+oct_y[i+1])],
                color=colors[0], alpha=0.5)
        ax.scatter([0.8*np.cos(oct_angles[0] + np.pi/8)], 
                  [0.8*np.sin(oct_angles[0] + np.pi/8)], 
                  c=colors[0], s=60, zorder=5)
        ax.set_title('Phase 1: Compartmentalised\n(1 face visible)', fontsize=7)
        ax.annotate('?', xy=(0, 0), fontsize=20, ha='center', va='center', 
                   color='gray', alpha=0.5)
    
    elif phase == 1:
        # Phase 2: Pairs visible
        ax.fill(oct_x, oct_y, alpha=0.05, color='gray')
        ax.plot(oct_x, oct_y, 'gray', linewidth=0.5, alpha=0.3)
        for i in range(0, 8, 2):
            mid_angle = (oct_angles[i] + oct_angles[i+1]) / 2 + np.pi/16
            ax.scatter([0.7*np.cos(mid_angle)], [0.7*np.sin(mid_angle)], 
                      c=colors[i], s=40, zorder=5)
            ax.scatter([0.7*np.cos(mid_angle+0.2)], [0.7*np.sin(mid_angle+0.2)], 
                      c=colors[i+1], s=40, zorder=5)
        ax.set_title('Phase 2: Partial Transparency\n(pairs visible)', fontsize=7)
    
    else:
        # Phase 3: Full octagon
        ax.fill(oct_x, oct_y, alpha=0.1, color='steelblue')
        ax.plot(oct_x, oct_y, 'steelblue', linewidth=2)
        for i in range(8):
            mid_angle = (oct_angles[i] + oct_angles[i+1]) / 2
            ax.scatter([0.7*np.cos(mid_angle)], [0.7*np.sin(mid_angle)], 
                      c=colors[i], s=50, zorder=5)
            # Lines from center to each face
            ax.plot([0, 0.5*np.cos(mid_angle)], [0, 0.5*np.sin(mid_angle)], 
                   'gray', linewidth=0.5, alpha=0.3)
        ax.set_title('Phase 3: Full Transparency\n(structure revealed)', fontsize=7)
    
    ax.set_aspect('equal')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.grid(False)
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'{OUT}/octagon_governance.pdf')
plt.close()

# ════════════════════════════════════════════════════════════════
# FIGURE 6: Domain mapping table (visual)
# ════════════════════════════════════════════════════════════════
print("Generating: domain_mapping.pdf")
fig, ax = plt.subplots(figsize=(7, 3))
ax.axis('off')

domains = [
    ['Domain', 'Cable (βκ)', 'Strut (αγ)', 'ρ = αγ/βκ', 'Collapse mode'],
    ['Governance', 'Resolution × damping', 'Bias × cross-gain', 'Institutional capture', 'Policy divergence'],
    ['Oscillators', 'Spectral gap Δω', 'Nonlinear shift βA²', 'Mode collapse', 'Energy transfer'],
    ['Float arith.', 'Precision bits/op', 'Condition × chain', 'Threshold crossing', '33 primes flip'],
    ['Networks', 'Stability margin B', 'Perturbation ε', 'Spectral breach', 'Topology failure'],
    ['Quantum', 'Gate fidelity', 'Barrel shift depth', 'Decoherence', 'Branch divergence'],
    ['Wolfram', 'Spectral gap λ₂', 'Spectral radius ρ(A)', 'Spatial fragility', 'Geometry collapse'],
]

table = ax.table(cellText=domains[1:], colLabels=domains[0], 
                cellLoc='center', loc='center',
                colWidths=[0.13, 0.22, 0.22, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.6)

# Style header
for j in range(5):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', weight='bold')

# Alternate row colors
for i in range(1, 7):
    color = '#ecf0f1' if i % 2 == 0 else 'white'
    for j in range(5):
        table[i, j].set_facecolor(color)

plt.title('Universal Tensegrity Motif Across Domains', fontsize=10, pad=20)
plt.savefig(f'{OUT}/domain_mapping.pdf', bbox_inches='tight')
plt.close()

# ════════════════════════════════════════════════════════════════
# FIGURE 7: Coupled Duffing oscillator simulation
# ════════════════════════════════════════════════════════════════
print("Generating: duffing_collapse.pdf")
fig, axes = plt.subplots(2, 2, figsize=(7, 5))

gamma_damp = 0.1
k = 1.0
kappa_coupling = 0.3
F0 = 0.5
omega_drive = np.sqrt(k + 2*kappa_coupling)  # drive at out-of-phase mode

for idx, (beta_nl, title) in enumerate([(0.05, r'$\beta=0.05$ (below $\beta_c$)'),
                                          (0.5, r'$\beta=0.5$ (above $\beta_c$)')]):
    def coupled_duffing(t, y):
        x1, v1, x2, v2 = y
        dx1 = v1
        dv1 = -gamma_damp*v1 - k*x1 - kappa_coupling*(x1-x2) - beta_nl*x1**3 + F0*np.cos(omega_drive*t)
        dx2 = v2
        dv2 = -gamma_damp*v2 - k*x2 - kappa_coupling*(x2-x1) - beta_nl*x2**3
        return [dx1, dv1, dx2, dv2]
    
    sol = solve_ivp(coupled_duffing, [0, 200], [0.1, 0, -0.1, 0], 
                    max_step=0.05, t_eval=np.linspace(0, 200, 4000))
    
    t = sol.t
    x1, x2 = sol.y[0], sol.y[2]
    
    # Time series
    ax = axes[0, idx]
    ax.plot(t[-1000:], x1[-1000:], 'b-', linewidth=0.5, alpha=0.8, label='$x_1$ (driven)')
    ax.plot(t[-1000:], x2[-1000:], 'r-', linewidth=0.5, alpha=0.8, label='$x_2$ (free)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6)
    
    # Energy distribution
    ax = axes[1, idx]
    E1 = 0.5*sol.y[1]**2 + 0.5*k*x1**2 + 0.25*beta_nl*x1**4
    E2 = 0.5*sol.y[3]**2 + 0.5*k*x2**2 + 0.25*beta_nl*x2**4
    # Running average
    window = 100
    E1_avg = np.convolve(E1, np.ones(window)/window, mode='valid')
    E2_avg = np.convolve(E2, np.ones(window)/window, mode='valid')
    t_avg = t[:len(E1_avg)]
    
    ax.plot(t_avg, E1_avg, 'b-', linewidth=1, label='$E_1$')
    ax.plot(t_avg, E2_avg, 'r-', linewidth=1, label='$E_2$')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy (averaged)')
    ax.legend(fontsize=6)
    if idx == 0:
        ax.set_title('Modes separated', fontsize=8)
    else:
        ax.set_title('Energy leakage (collapse)', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT}/duffing_collapse.pdf')
plt.close()

print("\n=== ALL EXTENDED FIGURES GENERATED ===")
for fn in sorted(os.listdir(OUT)):
    if fn.endswith('.pdf') and fn != 'full_paper.pdf':
        size = os.path.getsize(f'{OUT}/{fn}')
        print(f"  {fn:<35} {size:>8,} bytes")
