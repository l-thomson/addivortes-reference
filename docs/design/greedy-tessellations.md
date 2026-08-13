# Greedy tessellation ensembles: does the XGBoost pattern transfer?

*Design note. Nothing here is shipped; `examples/greedy_prototype.rs` is the
working prototype the numbers come from.*

Two questions get asked whenever someone meets AddiVortes having come from
gradient boosting:

1. Could you build "XGBoost, but with Voronoi tessellations"?
2. Is there a non-Bayesian AddiVortes — a random forest, but with
   tessellations?

Both are buildable, both are prototyped, and they get different answers:

- **Boosting: yes.** Once the greedy search is pointed at the right thing it
  matches the MCMC's accuracy on Friedman #1 in roughly a third of the time.
  Pointed at the wrong thing — which is what a literal port of XGBoost does —
  it loses by a wide margin. That contrast is the most useful thing in this
  note.
- **Bagging: no, and instructively so.** The reason it fails says something
  about what a Voronoi base learner is good for.

## The family tree is not bagging → boosting

The framing "AddiVortes is to XGBoost as bagging is to boosting" is the one
thing to correct before anything else, because it points the design at the
wrong axis.

| | Members combine by | Members are | Fitted by |
|---|---|---|---|
| Random forest | averaging | deep, low-bias, decorrelated | independent greedy fits on bootstrap resamples |
| XGBoost | summing | shallow, high-bias | greedy stagewise second-order optimisation |
| BART / AddiVortes | summing | shallow, high-bias | MCMC over structures; priors regularise |

AddiVortes is already a sum of weak learners. It is in the same *function
class* as boosting — `F(x) = Σ_j g(x | T_j, M_j)` is precisely a boosted
ensemble's functional form. What separates it from XGBoost is not how members
combine but **how the ensemble is fitted**: a posterior explored by backfitting
MCMC, with the cell-count and payload priors doing the regularising, versus a
greedy stagewise point estimate with explicit `λ`, `γ`, `η` and early stopping.

So question 1 is "swap the fitting principle, keep the structure", and question
2 is a genuinely different third model. Worth keeping them apart.

## What transfers exactly: the gain algebra

Take the standard second-order objective over one tessellation with cells
`k = 1..b`:

```text
L = Σ_i [ g_i·μ_{c(i)} + ½ h_i·μ_{c(i)}² ] + γ·b + ½λ Σ_k μ_k²
```

with `g_i`, `h_i` the first and second derivatives of the loss at the current
fit. Nothing in this depends on the partition being a tree, so all of XGBoost's
closed forms survive verbatim:

```text
μ*_k = −G_k / (H_k + λ)          score_k = −½ · G_k² / (H_k + λ)
```

where `G_k = Σ_{i∈k} g_i` and `H_k = Σ_{i∈k} h_i`. Loss-agnostic, exactly as in
trees: squared error, logistic, Poisson, quantile, ranking — change two lines.

## The Voronoi geometry is kinder than expected

The part that *is* model-specific is working out which observations move when
the structure changes, and here Voronoi assignment has a property worth
knowing:

> Adding a centre never changes the distance from any point to any existing
> centre. So the observations that move are exactly
> `{ i : d(x_i, c_new)² < best_key_i }`, and every one of them lands in the new
> cell.

Adding a centre is therefore monotone — it only *steals* — and its gain is one
`O(n)` pass given the cached nearest-centre distance per row:

```text
gain = ½ [ G_new²/(H_new+λ)
         + Σ_k ( (G_k−G_k^out)²/(H_k−H_k^out+λ) − G_k²/(H_k+λ) ) ] − γ
```

Two things follow. First, the crate already maintains exactly the cache this
needs: `AssignmentCache::best_keys` is the per-row nearest-centre distance the
MCMC's incremental reassignment uses, so a greedy grower reuses the engine's
existing hot-path machinery rather than needing new machinery. Second, note the
shape of the split: a tree splits **one parent into two children**, while
adding a centre takes a slice out of **every** cell at once. A `b`-cell
tessellation in a `d`-dimensional subspace is a `b`-way *oblique* partition —
every cell boundary is a perpendicular bisector hyperplane — which is far more
expressive per unit of structure than an axis-aligned stump.

## What does not transfer: exhaustive split enumeration

This is the crux, and it is worth being blunt about it, because it is the one
thing that makes boosted trees what they are.

In a tree the candidate set is **finite and small**: `p` features ×
at most `n−1` thresholds. Pre-sort each feature once and a single prefix-sum
sweep gives the *exact* argmax split in `O(n)` per feature. Almost everything
XGBoost and LightGBM are famous for — histogram binning, the approximate
quantile sketch, sparsity-aware split finding, the cache-blocked column layout,
GOSS — is engineering *around that enumeration*.

A Voronoi centre is a point in a continuous `d`-dimensional subspace. There is
no enumeration to be exact or approximate about. The greedy step therefore
becomes **best-of-K over sampled candidate centres**, which is a randomised
search, not an argmax. Consequences, in descending order of importance:

- **A sampling budget `K` appears where an exact answer used to be**, and there
  is no setting of it at which the search becomes exact. The obvious move is to
  turn `K` up until the search is "good enough"; the next section is about why
  that is the wrong instinct.
- **Training is stochastic.** A boosted-tree fit is deterministic given the
  data (modulo subsampling); here the seed matters roughly as much as it does
  in the MCMC. One of the practical selling points of GBMs quietly goes away.
- **Cost per unit of structure is much worse.** After an `O(n·K·d)` distance
  precompute, each added centre costs `O(n·K)` and yields **one** new cell. A
  tree spends `O(n·p)` per *level* and yields `2^level` cells. Trees buy
  partition granularity exponentially; tessellations buy it linearly.

That last point is the whole cost story, and it answers question 2.

## The literal port puts its search budget in the wrong place

This is the finding worth carrying away, and it is not the one the cost
analysis above predicts.

XGBoost's greedy step searches **(feature, threshold) jointly and
exhaustively**. Port it literally to tessellations and you keep the "search
positions hard" half — many candidate centres, take the best gain — while the
"search features" half quietly disappears, because a tessellation's subspace is
just *drawn at random* the way the paper's prior draws it. The prototype's
first version did exactly this, and it lost to the MCMC by a mile.

The fix is to spend the budget the other way round: draw several independent
subspaces per round, grow in each, keep whichever achieved the most gain
(`subspace_tries`), and cut the candidate pool right back. Holding the fit time
roughly constant, moving budget from centres to subspaces is worth about
0.25 RMSE on this benchmark — the difference between clearly losing to the
MCMC and slightly beating it.

From the tuning sweep, 3000 rounds, 8 cells, `η = 0.1` (these are sweep
numbers, stopped on the reported set — the benchmark below uses a separate
validation set for stopping, so read this table for its *shape*, not its
levels):

| candidates | subspace tries | RMSE | fit (s) |
|---:|---:|---:|---:|
| 5 | 1 | 1.614 | 0.2 |
| 20 | 1 | 1.568 | 1.0 |
| 40 | 1 | 1.686 | 2.1 |
| 80 | 1 | 1.649 | 4.4 |
| 10 | 3 | 1.400 | 1.3 |
| 40 | 3 | 1.444 | 6.0 |
| **5** | **8** | **1.375** | **1.2** |
| 20 | 8 | 1.362 | 7.4 |
| 80 | 8 | 1.412 | 33.8 |

Read down the `tries = 1` block and the candidate pool buys almost nothing —
80 candidates is *worse* than 20, at twenty times the cost. Read across to
`tries = 8` and every row improves by more than the whole candidate column
spans. Five candidates with eight subspaces beats eighty candidates with one,
and does it about four times faster.

Why it works out this way is worth stating plainly, because it is a property of
the model rather than of this benchmark:

- **Which covariates a tessellation lives in is the high-value discrete
  choice.** Five of the ten columns here are pure noise, so a random 4-column
  subspace is usually mostly noise, and no amount of clever centre placement
  rescues a tessellation fitted in the wrong subspace.
- **Where the centres go inside a good subspace is low-value.** A handful of
  centres partitions a smooth function about as well wherever they land, so the
  marginal return on the 40th candidate is close to zero — and past a point it
  is *negative*, because a harder-optimised base learner is a less diverse one,
  which is the ordinary boosting trade-off.

And note what the MCMC was already doing: the sampler's add-dimension,
remove-dimension and swap moves are accepted on the likelihood, so it has been
searching subspaces all along, even though it proposes them at random. The
naive greedy port simply dropped that half of the search. Read that way the
corrected design is less "XGBoost for AddiVortes" than "the sampler's
dimension moves, made greedy".

## Question 2: the random-forest analogue

It is well defined and falls straight out of the same grower: bootstrap the
rows, take a random subspace, grow a tessellation by variance reduction, average
the members instead of summing them. Setting `g_i = −y_i`, `h_i = 1`, `λ → 0`
in the gain algebra above recovers exactly this — the payload becomes the cell
mean and the gain becomes variance reduction — so the prototype implements it
by changing only the working response.

It also *does not work very well*, and the linear-vs-exponential cost point
above says why before you run it. Bagging's whole mechanism is averaging away
the variance of **deep, low-bias** members: a random forest grows each tree to
near-purity, hundreds of leaves, and relies on decorrelation to fix the
resulting overfit. Depth is exactly what is expensive for tessellations. A tree
reaching 512 leaves costs about nine `O(n·p)` passes; a tessellation reaching
512 cells costs 512 `O(n·K)` passes. So the bagged version is forced to run
with far too few cells per member, each member is badly underfit rather than
overfit, and averaging underfit members does not help — averaging only removes
variance, and underfitting is bias.

Subspace search helps here too — it takes the bagged version from 3.87 to
2.83 — but it cannot rescue it, and it costs 20 seconds to do so. The
prototype also shows the depth squeeze directly: given a 60-cell cap, the
greedy grower stops at 29, because the minimum-cell-occupancy guard —
the counterpart of the sampler's empty-cell rejection — blocks further
additions once cells get small. A tree in a random forest hits `min_samples_leaf
= 1` and keeps going; a tessellation cannot, because every added centre carves
from *all* existing cells at once and drives several of them under the floor
simultaneously. The very property that makes a centre cheap to score is what
stops the partition getting deep.

Boosting has the opposite requirement — many **shallow, cheap** members, each
correcting the last — and a small Voronoi tessellation is an unusually strong
shallow learner because of the `b`-way oblique property above. Notice that the
tuned boosting config went the other way entirely: **4 cells**, `η = 0.05`, and
a candidate pool of 5. The best members are nearly the weakest ones available.
**The base learner suits boosting and not bagging**, which is a reasonable
guess at why the published method is additive rather than bagged in the first
place.

The literature agrees by omission: there is no established "Voronoi random
forest". What exists nearby is
[Super-k](https://arxiv.org/pdf/2012.15492) (a single piecewise-linear Voronoi
classifier), [self-organising hierarchical Voronoi
classifiers](https://www.sciencedirect.com/science/article/abs/pii/S0020025522004017)
(prototype-based, recursive), and the Mondrian / random-tessellation forest
line, which partitions with *hierarchical hyperplane cuts* rather than
nearest-centre assignment — recovering exactly the recursive structure that
makes depth cheap.

## Measured

Friedman #1, `n_train = 500`, `p = 10` (five signal columns, five pure noise),
`σ = 1`, so test RMSE 1.00 is the irreducible floor. Three independent draws:
train, a validation set that owns the early-stopping decision, and a test set
that is only ever reported. Everything below is one run of
`cargo run --release --example greedy_prototype`.

| model | test RMSE | fit (s) | |
|---|---:|---:|---|
| AddiVortes (MCMC, m = 200) | 1.415 | 3.9 | 90% PI covers 88.1% of test rows |
| greedy boosting, literal XGBoost port | 1.677 | 2.2 | 40 candidates, 1 subspace |
| **greedy boosting, budget re-pointed** | **1.351** | **1.2** | 5 candidates, 8 subspaces |
| bagged greedy (RF analogue) | 2.826 | 20.7 | 200 members, 29 cells each |
| bagged random (extra-trees analogue) | 3.804 | 0.3 | 200 members, 60 cells each |

Single seed, single dataset, and the hyperparameters were chosen by a sweep, so
treat the third-decimal ordering between the top two rows as noise. The
robust claims are the gaps: the re-pointed boosting fit lands in the MCMC's
neighbourhood at roughly a third of the wall clock, the literal port does
not get close, and the bagged variants are not in the conversation.

Two caveats on the boosting row worth carrying: its held-out error was still
falling at round 2999 of 3000, so that figure is a ceiling rather than a
converged value; and every greedy row bought its speed by giving up the
interval column entirely.

## Checking the explanation, not just the result

The story above says the subspace advantage exists *because* most random
subspaces miss the signal columns. That is a mechanism claim, and it predicts
something specific: the advantage should collapse when there are no noise
columns to miss. Varying `p` with the five signal columns held fixed, otherwise
tuned settings, `tries = 1` versus `tries = 8`:

| p | noise columns | tries = 1 | tries = 8 | gap |
|---:|---:|---:|---:|---:|
| 5 | 0 | 1.376 | 1.294 | 0.082 |
| 10 | 5 | 1.638 | 1.351 | 0.287 |
| 15 | 10 | 1.825 | 1.548 | 0.277 |
| 25 | 20 | 2.022 | 1.774 | 0.249 |
| 40 | 35 | 2.138 | 1.870 | 0.268 |

Half the prediction holds and half does not, and the failure is the more
useful half.

The collapse is clear: with no noise columns the advantage drops to a third of
its size, which is what the mechanism requires. (It does not vanish entirely,
and should not — with `max_dims = 4` out of 5 columns there is still a subspace
*choice*, just no catastrophically wrong one to make.)

But the advantage does **not** keep growing. It jumps at the first noise
columns and then flattens at around 0.27 no matter how much noise is added.
The reason is visible in the other two columns: past `p = 10` *both* arms
degrade together. A fixed budget of eight subspaces is ample when 50% of
columns carry signal and hopeless when 12% do, so the `tries = 8` arm starts
missing the signal too. The advantage saturates not because subspace search
stops mattering but because a constant amount of it stops being enough.

Which is a concrete design consequence: **`subspace_tries` should scale with
`p`, not be a constant.** Whatever the right rule is — proportional to `p`, or
to `p / max_dims`, or adaptive on the achieved gain — a fixed default will
quietly underperform on wide data, which is exactly where a practitioner would
reach for this over the MCMC.

Standing caveat: one synthetic function, one seed per cell of that table. The
mechanism check is what makes the explanation credible, not the sample size.

## What you give up

| | AddiVortes (MCMC) | Greedy boosting |
|---|---|---|
| Point prediction | yes | yes |
| Credible / prediction intervals | yes, calibrated | no |
| Variable importance | posterior inclusion proportions | gain-based heuristic |
| Regularisation | priors, integrated out | `λ`, `γ`, `η`, early stopping — all tuned |
| Determinism | bit-exact per seed | seed-dependent search |
| Loss functions | conjugate/augmentable families | any twice-differentiable loss |
| Sample weights | via the weighted shelf entries | native |
| Cost model | sweeps × m × n | rounds × tries × cells × n × K |

The uncertainty row is the one that matters. Calibrated intervals are the
headline feature of this crate, and greedy fitting produces a point estimate
and nothing else. Anything that wants both is back to conformal wrappers or
bootstrapping the whole fit, which costs more than the MCMC did.

## Recommendation

**Worth building — but the accuracy result is not the reason.** The re-pointed
fit landing slightly ahead of the MCMC on one dataset at one seed is a tie, not
a win, and the MCMC was never slow enough for speed alone to matter at
`n = 500`. What the result does establish is that a greedy fitter is *not a
downgrade*: it stays in the same accuracy neighbourhood, which is the
precondition for wanting it at all. The actual case is the things the sampler
structurally cannot do:

- **Scale.** MCMC cost is sweeps × m × n with no early exit, and the sampler
  has no notion of stopping when it stops improving. A greedy fit stops when
  validation error turns, which is where the interesting `n` lives.
- **Arbitrary losses.** Quantile, ranking, Tweedie, custom business losses —
  anything with a second derivative, no conjugacy or augmentation needed.
- **Iteration speed.** Sub-second refits make hyperparameter search, feature
  screening and CV loops practical.
- **Warm starts and online updates.** Add rounds to an existing fit; a chain
  cannot be extended the same way.

And it comes with a cost the table above understates: the greedy fit has five
hyperparameters that genuinely need tuning (`candidates`, `subspace_tries`,
`max_cells`, `eta`, `lambda`), and the sweep that found them was itself more
expensive than every MCMC fit in this note combined. The MCMC needed a seed.

Three notes on how, if it gets built:

1. **Not a shelf entry.** All ten extension points sit *inside* the sampler;
   this replaces the sampler. It belongs in a separate crate or a feature-gated
   module reusing `Data`, `Tessellation`, the distance shelf and the scaler —
   the prototype already compiles against the public API only, which is the
   evidence that the seam is in the right place.
2. **The hybrid is more interesting than the replacement.** Use a greedy fit as
   the *initialiser* for the MCMC. You keep the calibrated intervals and start
   the chain somewhere sensible instead of at the prior, which should cut
   burn-in. That is cheap to try and does not fork the model.
3. **Resist the differentiable-Voronoi detour.** The objective is
   differentiable almost everywhere in the centre coordinates, so centres can
   in principle be moved by gradient descent instead of picked from a pool
   ([auto-differentiating the tessellation](https://arxiv.org/html/2312.16192v3)
   is a live line of work). It is the obvious way to fix "the search isn't
   exact" — and the measurements say the search over centre positions was never
   the binding constraint. Gradients would optimise the half that barely
   matters and cannot touch the half that does, because which columns a
   tessellation lives in is discrete. Worth knowing about; not the first thing
   to build.

## Sources

- [Super-k: A Piecewise Linear Classifier Based on Voronoi Tessellations](https://arxiv.org/pdf/2012.15492)
- [Self-organizing Divisive Hierarchical Voronoi Tessellation-based classifier](https://www.sciencedirect.com/science/article/abs/pii/S0020025522004017)
- [Binary AddiVortes](https://arxiv.org/html/2503.21792)
- [A Method for Auto-Differentiation of the Voronoi Tessellation](https://arxiv.org/html/2312.16192v3)
