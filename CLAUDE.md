# RGC shopper simulation: a learning project

## Who I am and why this exists
- I'm Anisia, a PhD researcher at Imperial (AI4Health CDT), building a digital twin for
  infection management from NHS EHR data. Comfortable with Python, pandas, SQL, PyTorch.
- I'm interviewing with Really Good Culture (RGC) for a 6-month **Simulation & Behavioural
  Science Lead** placement. The job description is in `context/SimulationBehaviourialLead_JD.pdf`
  and my notes are in `context/RGC_interview_prep.docx`. Read both at the start of a new session.
- **Primary goal: learning.** I want to become a better programmer and modeller, not just own a
  working simulation. Secondary goal: a small prototype I can explain confidently to RGC's founders.

## How to work with me (most important section)
- **Teach, don't just do.** Before writing any module, explain the idea in plain language,
  then the maths, then the code design. Keep explanations short and concrete.
- **I write the core logic myself.** For key functions (utility, softmax choice, habit update),
  write the function signature, docstring and a test, leave the body as `# TODO(Anisia)`, and
  give me hints. Only show a solution if I ask or I'm stuck after trying.
- **Small steps.** One module per step. Stop after each step and wait for me to say "continue".
- **Check my understanding.** At the end of each step, ask me 2 short questions an interviewer
  might ask about it.
- **Correct my terminology.** When I describe something imprecisely, give me the correct
  technical term (e.g. "multinomial logit", "calibration", "sensitivity analysis").
- **Explain trade-offs.** When there's a design choice, show me 2 options and why you'd pick one.
- **Log what I learn.** After each step, append 3–5 bullet points to `LEARNING_LOG.md`:
  concepts learned, terms to use, and one likely interview question with a good answer.
- Never make large unrequested changes. Ask before installing packages or deleting files.
- Commit after each working step, with a clear message. Ask me before committing.

## The model
A small, explainable **agent-based model (ABM)** structured around RGC's four simulation modules.

| Module | Implementation |
|---|---|
| **Environment** | ~6 products: `price`, `healthiness`, `novelty`, `brand_strength`, `treat_score`. One new product launches at week 4. |
| **People** | ~500 agents from 4 archetypes (budget, health-focused, treat seeker, habitual). Traits sampled from archetype-specific distributions: `price_sensitivity`, `health_orientation`, `novelty_seeking`, `habit_strength`. |
| **Intentions** | Each week each agent draws a **shopping mission**: `weekly_shop`, `top_up`, `treat`. Missions modify trait weights; on a treat occasion, price sensitivity is down-weighted (the JD's example). |
| **Extraneous factors** | Scenario switches: promotion (temporary price cut), price rise, health trend shock. |

**Choice model:** multinomial logit. For agent *i*, product *j*:
`U_ij = Σ_k (trait_ik × mission_weight_k × attribute_jk) + habit_bonus_ij + ε_ij`,
with choice probabilities via softmax over products. Use a temperature or scale parameter.

**Habit formation:** buying a product increases its habit bonus for that agent; the bonus
decays each week it isn't bought (reinforcement-style update).

**Simulation:** 26 weekly steps, fixed random seed, vectorised with numpy where it stays readable.

**Outputs:** market share over time; for the new product, **trial rate** (first purchases),
**repeat purchase rate** and time to repurchase; scenario comparison; a simple
**sensitivity analysis** on habit strength and price sensitivity.

**Honest framing:** parameters are set by assumption, not calibrated. Keep a `LIMITATIONS.md`
listing assumptions and how RGC data (Watch Humans, 28 behavioural dimensions, sales data)
would be used to **calibrate** and **validate** them (e.g. fit the choice model to purchase
histories, validate on held-out later time periods).

## Build plan (one step per session chunk)
0. **Plan:** propose the file structure, write out the utility equation, and justify parameter defaults.
1. **Config + environment:** a `dataclass` config and `products.py`. Learn: dataclasses, why there are no magic numbers.
2. **People:** `agents.py` with archetype sampling. Learn: sampling distributions, `np.random.default_rng`.
3. **Intentions:** `missions.py`. Learn: how context changes preferences.
4. **Choice:** `choice.py` (utility + softmax), I write it, with pytest tests. Learn: logit, softmax, numerical stability.
5. **Simulation loop + habit:** `simulate.py`. Learn: state updates and vectorisation versus loops.
6. **Scenarios + metrics:** `scenarios.py`, trial and repeat rates. Learn: scenario analysis.
7. **Sensitivity + demo:** `demo.ipynb` with 3–4 clear plots. Learn: which assumptions drive results.
8. **Wrap-up:** `LIMITATIONS.md`, and a 2-minute verbal explanation I can practise.

## Engineering rules
- Python 3.12; conda env `shopper-abm`. Libraries: numpy, pandas, matplotlib, pytest. No Mesa for now.
- Layout: `src/abm/` (products.py, agents.py, missions.py, choice.py, scenarios.py, simulate.py,
  config.py), `tests/`, `notebooks/demo.ipynb`, `context/` (ignored by git).
- Type hints and NumPy-style docstrings on every public function.
- Run tests with `pytest -q`. Tests must pass before committing.
- `.gitignore` must include `context/`. Never push to GitHub unless I ask.
