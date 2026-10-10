# Exact specification of the supplementary experiments

Version: 2026-09-16. Read with PROTOCOL.md and ADDENDUM_hub_correction.md. All parameters describe synthetic systems; they are not estimates of human behavior, content quality, or LLM capability.

## 1. Common population and graphs

N=1,000 and T=50. Initialization uses 450 opinions drawn from N(−0.5,0.2²), clipped to [−1,0]; 450 from N(0.5,0.2²), clipped to [0,1]; and 100 from N(0,0.1²), clipped to [−0.2,0.2]. The combined vector is shuffled. Susceptibility s_i is Beta(2,5), and disclosure sensitivity d_i is Uniform(0,1), independently drawn in the initialization stream. All initial attributes share that stream, so the exact generation order is part of the frozen implementation.

Graphs use the frozen NetworkX construction: BA attachment m=3; ER p=6/(N−1), with additional edges joining disconnected components to the original largest component; WS k=6 and rewiring p=0.1; SBM blocks defined by initial opinion sign. For SBM block sizes n0, n1, p_out=6/[5(n0−1)+n1] and p_in=5p_out. SBM node labels are mapped back to agent IDs. Neither SBM nor WS is explicitly forced to be connected. The graph builder's historical docstring claiming universal connectivity is therefore too strong.

Graph caches are keyed by topology, size, network seed and the SHA-256 of initial opinions. Graphs are read, never modified, after creation. Cache equality against uncached construction is tested for all four graph types.

## 2. E1/E2 frozen exposure and persuasion dynamics

The external target is z=0.6. A cumulative direct dose D is measured in **direct exposure events per agent**, including repeated exposures. At each active step, sample k distinct agents uniformly without replacement; agents may be selected again on later steps. For single-shot, all DN events occur at step 0. Burst distributes them over steps 0–4. Continuous distributes them over all 50 steps. Selected D values make these counts integral: 50, 500 or 950 total events for E1; 50 or 500 for E2. Equal direct-event totals do not imply equal unique reach, repeated exposure, or second-hand exposure.

For a directly seeded agent i, the proposed increment is

`g_i = b s_i (z−x_i) L_i + ε_i`,

where b is message strength, ε_i is N(0,(0.3b)²), and L_i=1 for unlabelled messages or max(0,1−λd_i) for labelled messages. The label attenuates the deterministic pull; it does not attenuate the additive noise. E1 strengths are 0.06,0.14,0.18. E2 uses 0.14. These numbers are not assigned empirical AI-quality interpretations.

A directly seeded agent activates when **|g_i|>0.01**, including negative increments that exceed the magnitude threshold. Each activated agent shares independently with each immediate neighbor with probability 0.3. A receiving neighbor j gets increment

`g_ij = 0.7 b s_j (z−x_j) L_j* + ε_ij`,

where ε_ij is N(0,(0.7×0.3b)²). L_j* is the same disclosure factor when labels are retained, and 1 when labels are lost on sharing. There is no recursive cascade within a step. All direct and second-hand pulls use the pre-step opinion vector. Multiple increments to one receiver are summed before applying `clip(x + direct + secondhand, −1, 1)`.

E1 additionally crosses μ=0 and 0.05, applying `x ← (1−μ)x + μx0` after every step, including steps without direct input. E2 uses μ=0. No exposure-dependent susceptibility decay is introduced in E1/E2.

### Inoculation selector and policy grid

For target budget fraction f, select floor(fN) agents, either randomly without replacement or by descending degree. The corrected hub selector uses **numeric node ID**, resolving equal degrees by ascending node ID. If compliance c<1, randomly choose floor(c×floor(fN)) of the targets. Multiply compliant agents' susceptibility by κ=0.2. Zero budget or zero compliance changes no agent.

The archived selector instead sorted positions in the graph-degree iteration and treated these positions as agent IDs. It also used max(1,...) at zero boundaries. Frozen code is preserved byte-for-byte; only the explicitly labelled corrected extension replaces this function. Equal-degree tie choices can differ outside SBM too. The archived test grid uses positive budget/compliance, so the boundary fix does not create a second change at those grid points.

E2 has 15 main policies: no intervention; λ∈{0.3,0.7} × retained/lost labels; strategy∈{random, hub} × budget∈{0.05,0.20} × compliance∈{0.5,1}; and retained λ=0.7 labels combined with either strategy at budget 0.05/compliance 1. The five policies involving hubs are also run with the archived selector, for 20 configurations total. Actual compliant counts for the inoculation grid are 25,50,100,200. No monetary cost or common cost conversion between labels and inoculation is assumed.

## 3. E1/E2 outcomes and pairing

Mean total shift: `N⁻¹ Σ_i |x_i(T)−x_i(0)|`.

Target-oriented signed shift: `N⁻¹ Σ_i [x_i(T)−x_i(0)] sign(0.6−x_i(0))`.

Conversion: the proportion with x_i(T)>0 among agents with x_i(0)<0, including initially negative centrist draws; it is not divided by the full population.

Time-averaged shift: `T⁻¹ Σ_(t=1)^T N⁻¹ Σ_i |x_i(t)−x_i(0)|`. This is a discrete post-update average, not total path length, and not the mean of absolute step increments.

Positive E2 benefit means the intervention lowers the specified outcome relative to no intervention. Positive hub advantage means `outcome(random)−outcome(hub)`. Positive label-retention advantage means `outcome(lost)−outcome(retained)`. These are changes in a simulated endpoint, not direct measurements of reduced harm.

Seeds use the fixed legacy namespace encoded by `NS` in `code/experiments.py`. Its hexadecimal representation preserves the original namespace bytes and exact random streams. A SHA-256 key combines suite, topology and replication; the first 15 hexadecimal digits form a master seed, and six separately hashed stream names produce network, initialization, seeding, sharing, noise and inoculation seeds. Within a suite/topology/replication, scenario and policy labels are excluded from the key. Across replication blocks the complete random vectors are independent. Within a block, branching can consume random streams differently, so paired seeds do not imply identical downstream events. Tables first average nuisance cells within each replication, then summarize the 30 independent block averages.

## 4. E3 independent peer-averaging extension

This is an explicitly specified extension inspired by the anchoring/peer-influence framework, not a reproduction of the original ABM or of a published model's entire specification. Foundational reference: Friedkin, N.E., & Johnsen, E.C. (1990). Social influence and opinions. The Journal of Mathematical Sociology,15(3–4),193–206. https://doi.org/10.1080/0022250X.1990.9990069.

Let W be the row-normalized adjacency in numeric node order 0,…, N−1. Add a self-loop for each isolate before normalization. At each step, let e_i(t)∈{0,1} mark direct exposure according to the same schedules. With stubbornness a∈{0.2,0.8}, coupling β∈{0.1,0.3}, and z=0.6:

`x_i(t+1) = a x_i(0) + (1−a)[(1−βe_i(t)) Σ_j W_ij x_j(t) + βe_i(t)z]`.

This convex update remains in [−1,1]. There is no additive persuasion noise, activation threshold, inoculation or label mechanism in this model. Every agent averages peers every step, including inactive exposure steps. Thus its absolute scale and parameters are not interchangeable with E1/E2.

For each topology/replication/a combination, run one no-input trajectory x_i^0(t), with e_i(t)=0, retaining the same graph and initial opinions. Main E3 outcomes replace `x(T)−x(0)` with `x(T)−x^0(T)` in total and signed shift; signed orientation still uses sign(0.6−x(0)). Conversion is exposed conversion minus control conversion among the initially negative agents. The trajectory outcome averages |x_i(t)−x_i^0(t)| over agents and time. Raw drift from initial opinions is separately retained and not substituted for input-attributable movement. Controls are reused within matched scenarios, not counted as independent controls for every run.

## 5. Factor allocations and uncertainty

Every complete-U game uses the uniform distribution over its own full Cartesian design. For a subset of factors S, project the centered vector of cell means onto the groups defined by S, then divide its squared norm by the number of cells. Shapley contributions average the resulting marginal subset contributions with standard factorial weights; interactions are allocated across factors. Signed finite-sample estimates are retained.

If H_j is the symmetric matrix for factor j and y_r is the vector of cell outcomes for independent replication block r,

`B_j, U = [R(R−1)]⁻¹ Σ_(r≠s) y_rᵀ H_j y_s`.

Its expectation is μᵀH_jμ under independent identically distributed blocks with finite second moments, even if cells within a block have correlated noise. With sample covariance S calculated using denominator R−1, this equals `ȳᵀH_jȳ−tr(H_jS)/R` exactly. A full covariance subtraction estimator is therefore algebraically identical. This is not a claim that complete-U always has lower RMSE or that its ratios are unbiased.

New-view shares divide B_j, U by the **same view's** positive total. E5 q divides a design's B_j, U by the corrected C-D50 total for that outcome; τ divides design total by corrected benchmark total; ρ divides by corrected benchmark B_j, U. Point ratios and bootstrap ratios with nonpositive denominators are suppressed. E5 tables count invalid ratio draws; any range based on remaining draws is conditional on that filter and is not a confidence interval. Tiny positive denominators can still make ratios unstable. Negative allocations are not clipped, and shares can exceed 1 or fall below 0 near zero signal.

For allocation sensitivity, sample multinomial replication multiplicities n_r and form cross-products only between **different original** block identities. The denominator is `(Σn_r)²−Σn_r²`; subtract `Σ n_r² y_rᵀH_jy_r` from the weighted quadratic numerator. Use 1,000 resamples for E1/E3/E5. E5 baseline and candidate share the first 30 count weights; extendedR=60 designs have a second independent 30-block stratum. The 2.5th and 97.5th percentiles are sensitivity ranges, not calibrated 95% confidence intervals.

For raw paired mean contrasts, report mean±t_(29,.975)×SE across 30 block averages. These quantify Monte Carlo precision in the specified simulator, are unadjusted for multiple comparisons, and do not quantify population or parameter-estimation uncertainty. They are not equivalence tests.

## 6. E4 known-truth experiment

On all 16 combinations of A, B, C, D∈{−1,+1}, set `μ=0.8A+0.5B+0.6AB+0.3C`. True Shapley allocations are(0.82,0.43,0.09,0), total 1.34. For each independent replication vector, draw 16-dimensional normal noise with covariance Σ. Four settings:

1. Independent homoskedastic: Σ=I.
2. Independent heteroskedastic: Σ=diag(σ²), σ=0.6+0.6×1(A=1)+0.4×1(B=1).
3. Heteroskedastic common shock: Σ=diag(σ)[0.4I+0.6×11ᵀ]diag(σ).
4. Heteroskedastic structured shock: v=(A+D)/√2; first form K=0.4I+0.6vvᵀ, normalize K to a correlation matrix C_K, and set Σ=diag(σ)C_Kdiag(σ).

Use R=10,30,60 and 1,000 datasets per covariance/R setting. Seed the sequential generator with 2026091604. Save covariances and dataset-level results. The first 500 datasets per setting receive 399 distinct-pair resamples and delete-one-block jackknife-normal intervals. Jackknife SE is sqrt[(R−1)/R × Σ_r(U_(−r)−mean U_(−r))²]; the interval uses the normal 1.96 quantile.

The independently implemented orthogonal-contrast estimator averages squared nonconstant factorial coefficients, splitting each interaction equally among its factors. It is verified against the projection-matrix code on correlated data, for both point estimation and bootstrap resampling. In-sample theoretical bias is tr(H_jΣ)/R. Bias MCSE uses the sample SD of estimates over 1,000 datasets divided by √1000. Coverage uses 500 datasets, with binomial MCSE and Wilson 95% intervals. No interval method is designated reliable solely because it performs favorably in one setting.

## 7. E5 provenance and timing

E5 was specified in a dated addendum after the node-ID defect was found and after E1–E3 executed, but before their aggregates were inspected. It is a disclosed implementation correction, not an externally preregistered experiment. All 40,320 archived hub rows are rerun with their original seeds and configurations; all 80,640 nonhub rows are retained. The corrected 120,960-row archive is a separately versioned replacement, not an enlarged independent dataset. Frozen input hashes and code snapshots remain available.
