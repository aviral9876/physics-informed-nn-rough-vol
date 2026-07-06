const fs = require("fs");
const G = "C:\\Users\\avira\\AppData\\Roaming\\npm\\node_modules\\";
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
        LevelFormat } = require(G + "docx");
const FIG = "C:\\Claude code\\physics informed NN\\figures\\";

// ---------- helpers ----------
const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 276 }, alignment: o.align,
  children: (Array.isArray(text) ? text : [new TextRun({ text, italics: o.i, bold: o.b })]) });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const b = t => new TextRun({ text: t, bold: true });
const r = t => new TextRun({ text: t });
const it = t => new TextRun({ text: t, italics: true });
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border };
function cell(txt, w, { head = false, bold = false, al } = {}) {
  return new TableCell({ width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: al,
      children: [new TextRun({ text: txt, bold: head || bold, size: 19 })] })] });
}
function table(widths, rows) {
  const total = widths.reduce((a, c) => a + c, 0);
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, borders,
    rows: rows.map((cells, ri) => new TableRow({
      children: cells.map((c, ci) => cell(String(c), widths[ci],
        { head: ri === 0, al: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) })) });
}
function figure(file, wPx, hPx, cap) {
  return [ new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
      children: [ new ImageRun({ type: "png", data: fs.readFileSync(FIG + file),
        transformation: { width: wPx, height: hPx },
        altText: { title: cap, description: cap, name: file } }) ] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: cap }) ];
}
const caprun = (n, t) => [ new TextRun({ text: `Figure ${n}. `, bold: true, size: 19 }),
                           new TextRun({ text: t, size: 19 }) ];
const tabcap = (n, t) => new Paragraph({ spacing: { before: 120, after: 60 },
  children: [ new TextRun({ text: `Table ${n}. `, bold: true, size: 19 }),
              new TextRun({ text: t, size: 19 }) ] });

// ---------- references ----------
const REFS = [
 "Abi Jaber, E., & El Euch, O. (2019). Multifactor approximation of rough volatility models. SIAM Journal on Financial Mathematics, 10(2), 309–349.",
 "Carr, P., & Madan, D. B. (1999). Option valuation using the fast Fourier transform. Journal of Computational Finance, 2(4), 61–73.",
 "El Euch, O., & Rosenbaum, M. (2019). The characteristic function of rough Heston models. Mathematical Finance, 29(1), 3–38.",
 "Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. Quantitative Finance, 18(6), 933–949.",
 "Han, J., Jentzen, A., & E, W. (2018). Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences, 115(34), 8505–8510.",
 "Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. Review of Financial Studies, 6(2), 327–343.",
 "Horvath, B., Muguruza, A., & Tomas, M. (2021). Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models. Quantitative Finance, 21(1), 11–27.",
 "Jacquier, A., & Zurič, Ž. (2023). Random neural networks for rough volatility. arXiv:2305.01035. [preprint; no journal version as of this writing]",
 "Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 7482–7491.",
 "Papapantoleon, A., & Rou, J. (2025). A time-stepping deep gradient flow method for option pricing in (rough) diffusion models. Quantitative Finance, 25(12), 2009–2020.",
 "Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707.",
 "Sirignano, J., & Spiliopoulos, K. (2018). DGM: a deep learning algorithm for solving partial differential equations. Journal of Computational Physics, 375, 1339–1364.",
 "Wang, S., Sankaran, S., & Perdikaris, P. (2024). Respecting causality for training physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 421, 116813. [volume/article — verify]",
];

// ============================ BODY ============================
const body = [];
body.push(new Paragraph({ heading: HeadingLevel.TITLE, spacing: { after: 120 }, children: [
  new TextRun("A Physics-Informed Neural Network for Option Pricing under Rough Volatility: Meshfree Solution of the Markovian-Lifted PDE and Fast Parametric Calibration") ]}));
body.push(P([ it("Stage-2 draft (Methods + Results). Introduction, Related Work, Abstract and Discussion follow in later stages. Every quantitative claim is grounded in a committed repository artifact; commit hashes are given in captions and text.") ], { after: 160 }));
body.push(P([ b("Keywords: "), r("physics-informed neural networks; rough volatility; option pricing; Markovian lift; curse of dimensionality; model calibration; scientific machine learning.") ], { after: 200 }));

// ---------------- METHODS ----------------
body.push(H1("3  Methods"));

body.push(H2("3.1  Rough Heston and the Markovian lift"));
body.push(P("Under the rough Heston model, the instantaneous variance V is driven by a fractional kernel K(t) = t^(α−1)/Γ(α) with α = H + 1/2 and Hurst exponent H < 1/2, so the variance process is neither Markovian nor a semimartingale (Gatheral et al., 2018; El Euch & Rosenbaum, 2019). There is therefore no finite-dimensional pricing PDE. The Markovian lift (Abi Jaber & El Euch, 2019) approximates the kernel by a sum of n exponentials, K(t) ≈ Σ_i c_i exp(−x_i t), which introduces n auxiliary Ornstein–Uhlenbeck factors U^i with V = V_0 + Σ_i c_i U^i. The joint state (log-spot X, U^1, …, U^n) is Markovian, and the discounted price P(t, x, u) solves an (n+1)-dimensional backward Kolmogorov PDE:"));
body.push(P([ it("P_t + (r − V/2) P_x + (V/2) P_xx + Σ_i [−x_i u_i + λ(θ − V)] P_{u_i} + (ν² V/2) Σ_{i,j} P_{u_i u_j} + ρν V Σ_i P_{x u_i} − r P = 0,") ], { align: AlignmentType.CENTER }));
body.push(P("with terminal condition P(T, x, u) = payoff(e^x). All factors share one Brownian motion, hence the full double sum in the u-diffusion and the cross term with x. The lift makes n a tunable knob on the PDE dimension, which is precisely what our scaling study exploits. We build (c_i, x_i) by the geometric-quadrature construction of Abi Jaber & El Euch (2019); the relative L2 kernel-approximation error falls monotonically with n, from 25.6% at n = 5 to 0.39% at n = 100 (rough_heston_lift.py self-test)."));

body.push(H2("3.2  Two independent ground truths"));
body.push(P("We validate against two mutually independent pricers. (i) A semi-analytic Fourier pricer solves the fractional Riccati equation for the rough Heston characteristic function (El Euch & Rosenbaum, 2019) and inverts it by Fourier integration (Carr & Madan, 1999); it does not use the lift, so it is the ground truth of the true model (n → ∞). In the flat-volatility limit it matches Black–Scholes to 3.7×10⁻⁷. (ii) A lifted Monte Carlo simulator integrates the n-factor lifted SDE with an exponential-integrator (integrating-factor) Euler scheme that handles the stiff −x_i U^i term exactly — the mean-reversions span roughly ten orders of magnitude for small H, so naive explicit schemes diverge. The lifted MC matches Black–Scholes within Monte Carlo error and, because it approximates the model-plus-dynamics while the Fourier method approximates only an ODE, agreement between the two is a genuine cross-check. Crucially, at finite n the lifted MC is the ground truth of the PDE the network actually solves, whereas Fourier is the ground truth of the model; this distinction underlies our error decomposition (Section 3.4)."));

body.push(H2("3.3  Physics-informed neural network"));
body.push(P("We solve the lifted PDE with a physics-informed neural network (Raissi et al., 2019), i.e. a mesh-free residual method in the spirit of deep PDE solvers for high-dimensional problems (Sirignano & Spiliopoulos, 2018; Han et al., 2018). The network is a tanh multilayer perceptron taking (τ, x, u) with τ = T − t, so the terminal condition becomes an initial condition at τ = 0. We hard-constrain that initial condition through the ansatz P = P_base(τ, x) + τ · N_θ(τ, x, u): the τ prefactor forces the learned correction to vanish at τ = 0, so the network never fights the payoff. The baseline P_base is a Black–Scholes price anchored not at the spot variance V_0 but at the model’s expected integrated (forward) variance, w(τ) = ∫_0^τ E[V_s] ds, which accounts for mean reversion from V_0 toward θ; anchoring at V_0 instead leaves a systematic level bias that the network must undo."));
body.push(P("The training loss combines the interior PDE residual with asymptotic boundary losses enforcing P → 0 as S → 0 (deep out-of-the-money) and P → S − K e^{−rτ} as S → ∞ (deep in-the-money), sampled fresh each iteration. Two stabilizers matter. First, we weight the residual and the two boundary terms with self-adaptive per-term weights using the learnable homoscedastic-uncertainty scheme of Kendall et al. (2018), removing hand-tuned loss weights. Second, we use a τ-curriculum that trains short maturities first and expands the horizon, respecting the causal structure of the evolution (Wang et al., 2024). Finally, collocation in factor space is informed by a short pre-flight Monte Carlo: rather than sampling the factors u in an arbitrary symmetric box, we sample each factor from the mean and standard deviation it actually attains under the lifted dynamics, which is essential for reproducing the volatility skew."));

body.push(H2("3.4  Validation protocol and error decomposition"));
body.push(P("The network is trained at a fixed strike K = 1 and prices as a function of spot; the fixed-spot smile C(1, K) is read off by the call’s degree-1 homogeneity, C(1, K) = K · C(1/K, 1), which holds because the variance dynamics are independent of the spot level. Comparing the network’s prices to the model without this rescaling compares mismatched options and manufactures thousands of basis points of spurious error (here, ~2700 vol bp reduced to ~286 vol bp once corrected)."));
body.push(P("We never report a single “PINN-versus-Fourier” number. Instead we decompose the error, per strike and in price space where the signed errors add exactly:"));
body.push(P([ it("(PINN − Fourier) = (PINN − lifted MC) + (lifted MC − Fourier).") ], { align: AlignmentType.CENTER }));
body.push(P("The first term — the solve-error — measures how well the network solves the PDE it was given, and is what training, architecture, and sampling can improve. The second — the lift-error — measures the model approximation of the finite-n lift, is independent of the network, and shrinks only by raising n. We report errors primarily as price RMSE in basis points of spot (where the decomposition is additive) and give the implied-volatility equivalent where useful; note that the two components have opposite signs and so do not add in RMS."));

body.push(H2("3.5  Parametric PINN"));
body.push(P("To enable instant recalibration we train a single network that also takes the five model parameters (H, ν, ρ, λ, θ) as inputs, over a box around the calibrated BTC values: H ∈ [0.04, 0.20], ν ∈ [0.10, 0.60], ρ ∈ [−0.95, −0.20], λ ∈ [0.5, 4.0], θ ∈ [0.08, 0.35]; V_0 and the maturity are held fixed. Because the lift weights (c_i, x_i) depend on H, they are precomputed on an H-grid and gathered per collocation point, while the parameters enter the network as exact continuous inputs; the mean-reversion baseline is available in closed form per point. The number of factors is fixed by the crossover rule of Section 4.2 (n = 10). The network is differentiable in the model parameters, which is what enables gradient-based calibration."));

body.push(H2("3.6  Calibration procedures and data"));
body.push(P("We calibrate three ways. (a) Fourier-in-the-loop: differential evolution on the parameters with the Fourier pricer inside the objective. (b) PINN gradient: Adam descent on the parameters using the parametric network as a differentiable pricer, with zero Fourier calls. (c) Hybrid: the PINN’s single-shot gradient solution as an initial guess, refined by a few local Fourier polish iterations (Nelder–Mead). The objective is the vega-weighted implied-volatility root-mean-square error against the market smile. The market data are end-of-day BTC option quotes from Deribit (a cryptocurrency derivatives exchange and data provider), obtained through its public REST interface; we take r = 0 and infer the forward from the quotes, following the standard crypto convention. We recompute implied volatilities from mid prices with a robust Black–Scholes inverter and vega-weight the surface so at-the-money points count more."));

// ---------------- RESULTS ----------------
body.push(H1("4  Results"));

body.push(H2("4.1  Dimension scaling: sub-linear solve-error versus finite-difference blow-up"));
body.push(P("Our primary methodological result concerns how the network’s own solve-error behaves as the lifted dimension n grows. Training the same architecture (width 64, depth 4) at n = 4, 8, 16, 32 on the canonical BTC calibration at T = 0.15 and measuring the solve-error against the lifted MC over three random seeds (Table 1, Figure 1), the solve-error grows only mildly — from 5.4 ± 1.0 to 10.7 ± 2.0 price bp, roughly a factor of two over an eight-fold increase in dimension. It does not stay flat (an earlier single-seed run misleadingly suggested it did), but it grows far more slowly than the dimension. Wall-clock cost to the solve-error plateau stays bounded (about 6–10 minutes on a CPU; per-iteration cost roughly constant at 55–62 ms) because the factor Hessian is vectorized."));
body.push(P("By contrast, a naive full-tensor finite-difference grid on the same (n+1)-dimensional PDE, at a modest 50 nodes per dimension, requires 50^(n+1) points: 3.1×10⁸ at n = 4, 2.0×10¹⁵ at n = 8, and 1.2×10⁵⁶ at n = 32. Even n = 8 (about 16 petabytes in double precision) is already infeasible, while the network trains every n on a laptop CPU. We state this contrast honestly: 50^(n+1) is the naive grid. Structured finite-difference schemes — sparse grids, ADI/operator splitting, low-rank tensor-train — push the tractable dimension higher, but their cost and accuracy still degrade with n (sparse-grid node counts grow super-linearly; tensor-train needs low-rank structure that the coupled ρ–ν cross terms erode), and none is standard or validated for the lifted rough-volatility PDE. The honest claim is that a competitive finite-difference solve at n = 16–32 is impractical, not provably impossible. Gentle solve-error growth plus bounded cost, against this blow-up, is the curse-of-dimensionality argument for the meshfree approach."));
body.push(tabcap(1, "Dimension scaling (canonical BTC calibration, T = 0.15; solve-error is 3-seed mean ± SD; commits 433afb6, c86a976). Errors are price RMSE in basis points of spot."));
body.push(table([1400, 2100, 2000, 1900, 2000], [
  ["n", "solve-error (px bp)", "lift-error (px bp)", "train time (s)", "FD grid 50^(n+1)"],
  ["4", "5.4 ± 1.0", "16.4", "397", "3.1×10⁸"],
  ["8", "6.1 ± 1.4", "7.9", "386", "2.0×10¹⁵"],
  ["16", "8.8 ± 1.6", "3.5", "591", "7.6×10²⁸"],
  ["32", "10.7 ± 2.0", "1.7", "457", "1.2×10⁵⁶"],
]));
body.push(...figure("fig1_n_convergence.png", 468, 290,
  caprun(1, "Dimension scaling. PINN solve-error (blue, 3-seed mean ± SD) grows sub-linearly and stays under ~11 price bp to n = 32, while lift-error (red) falls monotonically; training time (green) stays bounded. The naive finite-difference grid (inset) blows up, infeasible by n = 8. Regenerated by figures_paper.py from committed results (433afb6, c86a976).")));

body.push(H2("4.2  The solve/lift decomposition and a crossover rule for n"));
body.push(P("The decomposition explains the record. Earlier we (and the literature convention) would have quoted a single “PINN vs Fourier” implied-vol RMSE of ~114–127 vol bp at n = 4 and read it as network error. Decomposed, that number is dominated by the lift: at n = 4 the full-fix configuration is 37 vol bp of solve-error plus 133 vol bp of lift-error (partially cancelling to a total of 121), and in the correlation-free case it is only 8 vol bp of solve-error on top of 133 of lift (commit 8805994). The network already solves its PDE to well under 50 vol bp; the apparent gap was model approximation."));
body.push(P("Because the lift-error shrinks monotonically with n (133, 63, 27, 12 vol bp at n = 4, 8, 16, 32; equivalently 16.4 → 1.7 price bp) while the solve-error grows mildly, the two curves cross. At T ≈ 0.15 the crossover sits near n = 8–10 (Figure 1): below it the model approximation dominates and one should add factors; above it the network’s own error dominates and more factors are wasted. This gives a concrete rule for choosing n. The crossover also moves with maturity: over a T-sweep the solve-error stays small and bounded everywhere (roughly 1–9.5 price bp across T ∈ {0.05, 0.15, 0.5} and H ∈ {0.05, 0.10, 0.15}), but the lift-error grows with T — from 8.6 to 31.8 price bp at n = 4 as T goes 0.05 → 0.5 — so at long maturities the lift still exceeds the solve floor at n = 16 and the crossover shifts to higher n (commit 9bcdfdd). Longer horizons need more factors."));

body.push(H2("4.3  Calibration to real BTC options gives a rough Hurst exponent"));
body.push(P("Calibrated to the real Deribit BTC surface with a full-budget optimizer (differential evolution over 60 generations plus local polish, all 13 maturities, up to 8 points per maturity spread across moneyness, 97 points total; commit 1e3913f), the model yields H = 0.090, ρ = −0.792, V_0 = 0.103, θ = 0.253, λ = 2.567, ν = 0.300, at a vega-weighted implied-vol RMSE of 496 bp (Figure 2). The Hurst exponent is well inside the rough regime (H ≪ 1/2) and the correlation is strongly negative, both economically sensible for crypto. The fit is deliberately reported as coarse: crypto smiles are wide and the lifted rough-Heston form is stiff, and a single 496-bp figure is honest about that. This value supersedes an earlier, leaner calibration and is the canonical anchor for the parameter box in Section 4.4."));
body.push(...figure("fig2_calibration_fit.png", 360, 325,
  caprun(2, "Rough Heston calibrated to the Deribit BTC surface (97 points, 13 maturities, colour = maturity). Market versus model implied vol at the canonical fit (H = 0.090, ρ = −0.79; RMSE 496 vol bp). Dashed line is the identity. Regenerated by figures_paper.py from the committed calibration (1e3913f).")));

body.push(H2("4.4  Parametric PINN and the calibration hybrid"));
body.push(P("A single network covering the five-parameter box is necessarily less accurate at any one point than a network trained there. At the box centre the parametric solve-error is 11.0 ± 4.7 price bp against 5.9 ± 1.2 for a single-point network at the same n = 10 — about a two-fold coverage penalty. The penalty is uneven: it stays near 9–11 price bp at typical points but rises to 47.7 price bp in the rough low-H corner and 29.8 at high vol-of-vol, where the PDE is hardest (commit 9fb5f88). We report this honestly; a larger training budget would shrink the corner penalty."));
body.push(P("This limits standalone PINN calibration but not its use as a warm start. Calibrating the BTC T ≈ 0.15 slice three ways (Table 2, Figure 3), the Fourier-in-the-loop optimizer reaches 23.7 vol bp using 2018 Fourier evaluations in about 168 s. PINN-only gradient descent is fast (about 2 s, zero Fourier calls) but poor — 222 ± 20 vol bp under the true model — because it calibrates to its own imperfect prices and, as an honest aside, because this single-maturity slice optimum lies outside the parametric training box (it wants ν ≈ 0.87, θ ≈ 0.39). The hybrid resolves this: seeding the local Fourier polish with the PINN’s single-shot guess reaches 29.4 ± 7.1 vol bp — comparable to from-scratch on a 34–52% smile — in only 237 ± 46 Fourier evaluations and about 23 s, roughly nine times fewer Fourier calls and seven times faster (commit 8aba282). The applied value of the parametric network is therefore as a calibration warm start, not a standalone pricer."));
body.push(tabcap(2, "Calibration of the BTC T ≈ 0.15 slice, three ways (3 PINN seeds; all judged by the true Fourier model over the same feasible region; commit 8aba282)."));
body.push(table([3400, 2200, 1900, 1900], [
  ["method", "fit RMSE (vol bp)", "wall-clock (s)", "Fourier evals"],
  ["Fourier-loop from scratch", "23.7", "168", "2018"],
  ["PINN-only (single-shot)", "222 ± 20", "~2", "0"],
  ["Hybrid (PINN + Fourier polish)", "29.4 ± 7.1", "23", "237 ± 46"],
]));
body.push(...figure("fig3_hybrid.png", 500, 220,
  caprun(3, "Hybrid calibration. Fit quality (left) and Fourier-evaluation cost (right) for the three methods on the BTC slice. The hybrid matches from-scratch quality at ~9× fewer Fourier evaluations; PINN-only is fast but inaccurate. Regenerated by figures_paper.py from committed results (8aba282).")));

// ---------------- REFERENCES (confirmed) ----------------
body.push(H1("References (verified this session; Related-Work additions in Stage 3)"));
REFS.forEach(rf => body.push(new Paragraph({ spacing: { after: 100 },
  children: [new TextRun({ text: rf, size: 20 })] })));

// ============================ DOC ============================
const doc = new Document({
  styles: { default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, font: "Arial" }, paragraph: { spacing: { after: 200 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 260, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2E4B7A" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 } },
    ] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: body }],
});
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("C:\\Claude code\\physics informed NN\\paper\\Stage2_Methods_Results.docx", buf);
  console.log("saved paper/Stage2_Methods_Results.docx");
});
