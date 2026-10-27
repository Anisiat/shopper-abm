# Learning log

## Step 0: Plan (2026-10-27)
- **Random utility model (RUM):** utility = systematic part V (traits × attributes + habit) + random part ε. If ε follows a Gumbel distribution, choice probabilities are exactly the softmax of V: that is the **multinomial logit (MNL)**.
- **Taste coefficients (β):** the "weights" in the rules. Price, health and novelty each get one. **Interaction terms** (trait × attribute) are how different people value the same product differently, which is **preference heterogeneity**.
- **Scale is not identified:** multiplying all β by 2 is the same as halving the temperature τ. So fix τ = 1 for the model and use it only as a sensitivity knob. Utility units mean "1 unit ≈ 2.7× the odds".
- **Outside option (no-purchase alternative):** lets a price rise reduce category sales, not only shift share between products. It also makes "rejection" measurable.
- **Interview Q:** *"Why a logit and not just picking the highest-utility product?"*
  **A:** "People aren't perfectly predictable. The random term stands for everything I don't model, like mood or shelf position. Assuming it's Gumbel gives the softmax in closed form, so I get choice probabilities I can fit to purchase data by maximum likelihood. The scale of that noise sets how deterministic choices are."

## Step 1: Config, environment and people (2026-10-27)
- **Frozen dataclasses** (`@dataclass(frozen=True)`) give you typed, immutable settings. Use `field(default_factory=...)` for mutable defaults. A run can't silently change its own config, which helps **reproducibility**.
- **No magic numbers:** every assumption lives in `config.py` or the product catalogue. A scenario or **sensitivity analysis** is then just a different config, with no code changes.
- **Beta(a, b)** is the natural distribution for traits bounded in [0, 1]. Mean = a/(a+b), and a larger a+b gives a tighter spread. The traits are **archetype-conditional distributions**, a simple form of **hierarchical / mixture population**.
- **`np.random.default_rng(seed)`:** one generator passed explicitly to every function. That's reproducible and doesn't depend on global state. The legacy `np.random.seed` is global and easy to break.
- **Interview Q:** *"How would you create synthetic shoppers that look like real ones?"*
  **A:** "Start from archetypes with assumed trait distributions, as here. Then **calibrate** them to data, for example by fitting each archetype's distribution to the 28 behavioural dimensions, or by fitting a **latent class choice model** to purchase histories so the data reveals the archetypes. Finally, check the synthetic population's joint distributions against held-out real shoppers."
