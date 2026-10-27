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

## Step 3: Intentions / shopping missions (2026-10-27)
- **Context-dependent preferences:** the same person weighs attributes differently depending on the occasion. Here the mission multiplies the taste coefficients, an **interaction between context and preferences**. In choice modelling this is often called **situational** or **context effects**.
- **Inverse-CDF sampling:** draw u ~ U(0,1) and pick the first category whose cumulative probability exceeds u. Done in a vectorised way, it samples a different categorical distribution for every agent in one line. It's the same trick we'll use to sample products from softmax probabilities.
- **Fancy indexing:** `table[missions]` turns a small (missions × terms) lookup table into an (agents × terms) matrix with no loop.
- **i.i.d. vs Markov missions:** each week's mission is drawn independently (**i.i.d.**). A **Markov chain** (a transition matrix between missions) would capture routines, such as a top-up after a big shop. That's a listed limitation.
- **Interview Q:** *"How would you know the treat weight on price is 0.3 and not 0.7?"*
  **A:** "Right now it's a hypothesis from the literature. With data, I'd estimate it: fit the choice model with a price × treat-occasion interaction on purchase data where the occasion is known, for example from Watch Humans interviews or basket composition. The size of that interaction coefficient, and its confidence interval, is the answer. It's also a key parameter for the sensitivity analysis."

### Step 3: expanded check-your-understanding answers
- **Q: Why multiply the mission weight into the utility instead of adding a "treat bonus"?**
  **A:** "A multiplier changes *how much a motive matters*, and the person's own trait still scales it. On a treat trip a budget shopper's price coefficient drops from -2.4 to about -0.7, but they are still more price-sensitive than a health-focused shopper, so heterogeneity survives. An additive bonus would shift every shopper by the same amount and ignore who they are. The multiplier is also an **interaction term** (occasion × price), which I can estimate from data and test: 'is the price coefficient smaller on treat occasions?'"
- **Q: Missions are drawn independently each week. What does that miss, and how would you model it?**
  **A:** "It misses routines: a big shop is usually followed by a few top-ups, and trips follow the calendar (payday, weekends, holidays). I'd model it as a **Markov chain**, where this week's mission depends on last week's through a transition matrix per archetype. The next step up is covariates, such as a payday dummy or a seasonal effect. With purchase histories I'd estimate the transition probabilities directly by counting transitions. The simplification goes in `LIMITATIONS.md`."

## Step 4: Choice model (2026-10-27)
- **Multinomial logit pipeline:** `taste_coefficients` (β × trait × mission weight) → `utility` (matrix product plus habit) → `choice_probabilities` (softmax over products plus buy-nothing) → `sample_choice` (inverse-CDF draw). All 500 agents are processed at once with no Python loop over agents.
- **Softmax and numerical stability:** subtracting the row maximum before `exp` doesn't change the result, because only *differences* in utility matter (**location normalisation**), and it stops `exp(1000)` overflowing. `-inf` utility gives exactly 0 probability, which is how unavailable products are excluded.
- **Outside option:** "buy nothing" is pinned at utility 0, so every product's utility is measured relative to it. With it, a price rise can *lose* sales, not only move them between brands, and rejection becomes measurable.
- **Temperature and scale:** doubling every β is the same as halving T, so only one of them is identified. T = 1 is the model, and other values are for **sensitivity analysis**.
- **Shared β, personal traits:** one population-wide β per term, scaled by each person's trait. This is a **random coefficients (mixed logit)** style of model. `brand_strength` and `treat_score` have no trait, so everyone shares the coefficient (a limitation).
- **Interview Q: "Is β different for each archetype?"**
  **A:** "No, one β per motive for the whole population. Archetypes differ through their trait distributions, and the trait multiplies β. That keeps the parameter count low, so β can be estimated from data. If the data showed archetypes with genuinely different β, I'd move to a **latent class logit**, where each class has its own coefficients and class membership is estimated rather than assumed. I'd compare the two with held-out log-likelihood."
- **Interview Q: "Why include a 'buy nothing' option, and why fix its utility at 0?"**
  **A:** "Without it every shopper buys something every week, so total category sales can never change and a price rise could only redistribute share. Utility is only identified up to a constant, because adding the same number to every option leaves the probabilities unchanged. So I fix one option at 0, and every other utility is relative to not buying."

## Step 5: Simulation loop and habit (2026-10-27)
- **State, rules, time steps:** the state is the habit matrix `H` (agents × products). The rules are mission, choice and habit update. The time loop applies them week by week. `H` is the only thing that carries information from one week to the next, so the loop can't be vectorised over weeks.
- **Delta rule (Rescorla–Wagner):** `H ← H + α(bought − H)` moves H a fraction α of the way toward what just happened. Repeated buying approaches 1 exponentially (1 − (1−α)ᵗ), and unused habits decay as (1−α)ᵗ. With α = 0.3 the half-life is about 2 weeks. That is a **testable, quotable parameter**.
- **Feedback loop and emergence:** buying raises H, which raises utility, which makes buying again more likely. Repeat purchase is never coded directly.
- **One-hot encoding with a dropped column:** index with `choice` into an array that has an extra "nothing" column, then slice that column off. This handles "bought nothing" without any special case.
- **Pure functions:** `update_habit` returns a new array and never modifies its input, so a run can't be corrupted by hidden side effects.
- **Interview Q:** *"Your model produces repeat purchasing. How do you know it isn't just the shopper's fixed tastes?"*
  **A:** "In this model both exist, so I'd separate them. **Taste** is fixed attributes × traits. **State dependence** is the habit term. If I switch the habit term off (`β_habit = 0`), repeat rates should fall towards what taste alone predicts. The gap is the habit contribution. With real data this is the classic **heterogeneity vs state dependence** identification problem: you need panel data and a model with both terms, and you check that the habit coefficient stays significant once individual tastes are controlled for."

## Step 6: Scenarios and metrics (2026-10-27)
- **Scenario analysis / counterfactuals:** a scenario is a set of outside events (promotion, price rise, health trend) applied to an otherwise identical run. The effect of the event is the *difference* between the scenario and the baseline.
- **Common random numbers:** every run with the same seed uses the same random draws, because the number of draws per week doesn't depend on the scenario. So runs match exactly until the event starts, and differences are caused by the event, not by sampling noise. This is a **variance reduction** technique, and `test_scenarios_match_baseline_until_they_start` checks it.
- **Right-censoring:** late triers have less time to repeat, so a naive repeat rate is biased downwards. `repeat_metrics` only counts triers observed for the full window (8 weeks). Time to repurchase among repeaters is also biased short, so quote it with care.
- **Monte Carlo error vs model error:** across 20 seeds, kefir trial is 61.6% ± 2.0 and repeat rate 46.1% ± 2.4. Differences smaller than about 2–5 points between a single pair of runs are within sampling noise. More runs reduce Monte Carlo error but can't fix a wrong model.
- **Category gate:** a per-week probability of buying in the category at all (0.5), separating *whether to shop the category* from *which product*. It brought the kefir trial rate from 76% to 61%. The assumption goes in `LIMITATIONS.md`.
- **Finding to be honest about:** switching habit off (`β_habit = 0`) only lowers the kefir repeat rate from 48% to 45%, which is inside the noise. At current defaults, repeat buying is driven mostly by fixed tastes, not habit. Step 7's sensitivity analysis should test this.
- **Interview Q:** *"The model says a 25% kefir promotion raises its trial by 5 points. How much do you trust that?"*
  **A:** "I'd trust the direction and the mechanism, not the size. The size depends on the price coefficient, which is an assumption here. With RGC data I'd calibrate the price coefficient on past promotions, then validate by predicting a held-out promotion and comparing the simulated uplift with the observed one. I'd also report an interval from running many seeds, and from sampling the uncertain parameters, rather than a single number."

## Step 7: Sensitivity analysis and demo (2026-10-27)
- **One-at-a-time (OAT) sensitivity analysis:** change one parameter, hold the rest fixed, repeat over many seeds. It shows which assumptions the answers depend on, so it tells you what to calibrate first. Its weakness is that it misses **interactions** between parameters. A **global** method (Latin hypercube sampling, Sobol indices) varies them together.
- **Monte Carlo confidence interval:** the band is mean ± 1.96·SD/√n across seeds. It shrinks with more runs, but it only measures sampling noise, never model error.
- **Habit cuts both ways:** raising the habit coefficient from 0 to 12 lifts kefir repeat rate (44% → 51%) and cuts trial (67% → 34%). Habit is the barrier to entry for a new product and also the source of retention.
- **Non-linear response:** the damage from a price rise grows with price sensitivity and then flattens (about −10 points after the rise for β from −3 to −6), because a very price-sensitive shopper had already left the dearer brand.
- **Visualisation rules used:** colour follows the entity (a product has the same colour in every chart), the new product takes the lead colour, a legend is always present, and only the story's series gets a direct label. Weekly shares are noisy, so the label shows an average rather than the last week.
- **Interview Q:** *"Which parameter would you calibrate first, and how?"*
  **A:** "The ones the outputs are most sensitive to and that I know least about. Here that's habit strength, since it drives both trial and repeat, and the price coefficient, since it drives every price scenario. I'd fit the choice model by maximum likelihood on purchase histories, using promotions and price changes as natural experiments to identify the price coefficient, then check on a held-out period that the simulated repeat rates and promotion uplifts match what actually happened."
