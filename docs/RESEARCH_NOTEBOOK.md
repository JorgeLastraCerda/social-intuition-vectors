# Research Notebook — Warmth & Competence in LLM Hiring

> **How to use this document.**
> Read slowly. Every section has a mix of explanation and questions.
> When you see a `[ ]` blank, fill it in yourself before reading ahead.
> When you see a 💭 — that's an invitation to think before you continue.
> When you see a 🔬 — that's a hands-on task you can do right now in Python or R.
> When you see a ⚖️ — there's a genuine choice to make and good arguments on both sides.
> Come back to this after conversations with Emrecan or Stephen — it will feel different each time.

---

## Where we are right now

We've done the scaffolding. The repo exists. The papers are downloaded and read. The data from Carina is here. The method notes are written. What we have NOT done yet is produce a single line of actual experimental output.

That's fine — the next moves are the interesting ones.

The project has roughly four big decisions coming up, and they're all connected. Before jumping into code, working through these decisions carefully will save a lot of pain later.

---

## Part A — Understanding Carina's Data (Start Here)

Before building anything, you need to understand the benchmark you're building toward.
Open a Python session (or a Jupyter notebook) and load the data yourself.

```python
import pandas as pd

# Warmth and competence ratings for names
names_df = pd.read_csv("data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv")

# Callback rates from real correspondence studies
callback_df = pd.read_csv("data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv")

# Category-level ratings (disability, religion, etc.)
cats_df = pd.read_csv("data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/categories/categories.csv")
```

### Exploration 1 — The ratings

Run `names_df.head(10)` and `names_df.describe()`.

💭 Before you run it: what do you expect the average warmth score to be? Is it closer to 30, 50, or 70?

My guess: `[ ]`

What I actually found: `[ ]`

---

Now compute the mean warmth and competence per name:

```python
name_means = names_df.groupby("name")[["warm", "competent"]].mean().reset_index()
```

Run `name_means.sort_values("warm").head(10)` — the coldest-rated names.
Run `name_means.sort_values("warm").tail(10)` — the warmest-rated names.

💭 Do the patterns you see surprise you? What do you notice about which names cluster at each end?

What I noticed: `[ ]`

💭 Now look at the *range* of the warmth scores. Warmth goes from 0 to 100, but what's the actual range in the data?

Min warmth: `[ ]` | Max warmth: `[ ]` | Standard deviation: `[ ]`

This range matters for our paper. If all names cluster between 45 and 60, our probe won't have much to predict. If names span from 20 to 80, we have real signal.

---

### Exploration 2 — The main finding

Merge the two datasets and plot warmth vs. callback:

```python
# Get per-name mean ratings and merge with callback rates
name_means = names_df.groupby(["study", "name"])[["warm", "competent"]].mean().reset_index()
merged = name_means.merge(callback_df, on=["study", "name"])

# Correlation
print(merged[["warm", "competent", "callback"]].corr())
```

Fill in the correlations from the output:

| | warmth | competence | callback |
|---|---|---|---|
| warmth | 1.0 | `[ ]` | `[ ]` |
| competence | `[ ]` | 1.0 | `[ ]` |
| callback | `[ ]` | `[ ]` | 1.0 |

💭 Which is a stronger predictor of callback — warmth or competence? Does this match your intuition?

My intuition before looking: `[ ]`

What the data says: `[ ]`

💭 Notice that warmth and competence are correlated with each other. Why might that be? Is it a real psychological phenomenon, or a measurement artifact, or both?

My thought: `[ ]`

---

### Exploration 3 — The race gap

```python
# What's the average warmth rating for Black-associated vs White-associated names?
# You'll need to get race info from the published_data df
# Hint: look at what columns are in callback_df
print(callback_df.columns.tolist())
```

What columns are there? `[ ]`

Is race available? `[ ]`

If yes, compute:
- Mean callback for Black-associated names: `[ ]`
- Mean callback for White-associated names: `[ ]`
- Mean warmth rating for Black-associated names: `[ ]`
- Mean warmth rating for White-associated names: `[ ]`

💭 The paper's finding is that this warmth gap *mediates* the callback gap. In plain language, that means: the reason Black-associated names get fewer callbacks is (at least partly) because they're perceived as less warm. Do you find that convincing? What would an alternative explanation be?

My thoughts: `[ ]`

---

### Exploration 4 — Categories (the broader picture)

```python
# Compute mean warmth and competence per category level
cat_means = cats_df.groupby(["category", "level"])[["warm", "competence"]].mean().reset_index()
print(cat_means.sort_values("warm"))
```

💭 Which *types* of social signals (disability, religion, sexuality, etc.) show the most extreme warmth ratings?

What I found: `[ ]`

💭 This matters for our project because we need to decide which signals to include in our hiring prompts. The stronger the warmth/competence signal, the more likely our probe will pick it up. But there's also a counterargument: maybe we should include signals where the effect is subtle, to test the sensitivity of our probe. What do you think is the right strategy?

My preferred strategy: `[ ]`

---

## Part B — Decision 1: What Model?

We discussed this before, but now you can think about it more concretely.

### The options

**Option A: Llama 3.1 8B** (runs on a single L40 or even older GPU)
- Very well-studied in the mechanistic interpretability literature
- `transformer_lens` supports it natively, everything works out of the box
- Smaller models sometimes have noisier, less separable internal representations
- Good for fast iteration and debugging

**Option B: Gemma 3 27B** (needs 1 H100, ~54 GB)
- Bigger = cleaner internal directions (generally)
- Less studied in the interpretability literature, so our results are more novel
- Takes longer to run — every experiment is slower
- Stronger paper if the results are good

**Option C: Llama 3.3 70B** (needs 2 H100s, ~140 GB)
- Frontier-scale open-weights model, most impressive to publish with
- Risky: if the compute falls through, everything breaks
- Slower iteration cycle

### ⚖️ Your call

💭 Before you decide, think about this: the Anthropic paper used Claude Sonnet 4.5 — a model we can't access for probing. That means any open-weights model we use is already a departure. Does that argue for using the biggest possible model to compensate? Or does it argue for using a well-understood model so we can verify our method is working?

My reasoning: `[ ]`

My current choice: `[ ]`

Reason I might change my mind: `[ ]`

> **Suggestion:** Do the first full run on Llama 3.1 8B locally (or on a single L40). Once the pipeline works end-to-end and you've confirmed the probe is picking up warmth/competence correctly, switch to the larger model for the final results. This way you don't waste H100 time on debugging.

---

## Part C — Decision 2: How to Generate the Stories?

The concept stories are the foundation of everything. Bad stories → bad vectors → bad results. This step deserves real thought.

### What makes a good concept story?

A good high-warmth story should:
- [ ] Show warmth through actions, not labels (can't say "she was warm")
- [ ] Work across different topics (warmth in a hospital is different from warmth on a sports field)
- [ ] Be unambiguous (a neutral reader would agree it's warm)
- [ ] Not be correlated with confounds (warmth stories shouldn't always be set in families, etc.)

💭 What else would you add to this list? What could go wrong?

My additions: `[ ]`

### ⚖️ Who writes the stories?

**Option A: Use a frontier LLM (GPT-4o or Claude)**
- Fast, cheap, consistent
- Risk: the same model generates *and* later processes the stories — there could be circularity. A story that GPT-4o writes as "warm" might have subtle markers that Gemma/Llama picks up on for reasons other than genuine warmth.
- Mitigation: validate that the stories work on *human* raters, not just on the model.

**Option B: Human-written stories**
- Slower and more expensive (you'd need to write or crowdsource them)
- No circularity risk
- Small N might limit statistical power

**Option C: Hybrid**
- Use an LLM to generate drafts, then have humans revise and validate them
- Probably the right answer for a published paper, but slower

💭 The Anthropic paper used the model itself (Claude Sonnet 4.5) to write the stories and then validated by having researchers manually inspect a sample. Is that good enough for us? Does the fact that we're using a *different* model for generation vs. probing reduce the circularity concern?

My answer: `[ ]`

### Your first concrete task

Write a story generation prompt. Below is a skeleton — complete it:

```
You are a creative writer. Write a short paragraph (100–150 words) 
about the following topic: {topic}

The paragraph should describe a character who clearly demonstrates 
{high/low} {warmth/competence} through their actions and decisions.

Rules:
- Do not use the words: warm, cold, warmth, competent, incompetent, 
  skilled, or any direct label for the concept.
- Show the concept through specific behaviours, choices, or reactions.
- The paragraph should feel natural, not like a psychology exercise.

Topic: {topic}
Condition: {condition}

Write only the paragraph, nothing else.
```

💭 What's missing from this prompt? What would you add to make the stories better?

My revisions: `[ ]`

💭 How would you test that your prompt is working before generating all 4,800 stories? What's the minimum sample you'd check?

My answer: `[ ]`

### A topic list to start with

Here are 20 starting topics. Your task: add 10 more that you think will produce especially rich or diverse stories.

1. A team meeting where a decision needs to be made
2. Helping a stranger who dropped their groceries
3. Receiving critical feedback from a supervisor
4. Learning a new technical skill under pressure
5. A job performance review conversation
6. Responding to a customer complaint
7. Planning a group project with unclear roles
8. Discovering a colleague made a serious mistake
9. Presenting results that didn't go as expected
10. Mentoring a junior colleague on their first day
11. A difficult conversation with a family member
12. Dealing with an unexpected equipment failure
13. Preparing for an exam with limited time
14. Handling a billing dispute with a company
15. Coaching a sports team through a losing streak
16. Organising a community event alone
17. Negotiating a salary or contract
18. Working on a group assignment with a free-rider
19. Responding to an emergency at work
20. Being the only person who knows how to fix something

Your 10 additions:
21. `[ ]`
22. `[ ]`
23. `[ ]`
24. `[ ]`
25. `[ ]`
26. `[ ]`
27. `[ ]`
28. `[ ]`
29. `[ ]`
30. `[ ]`

---

## Part D — Decision 3: What Social Signals to Include?

Our hiring prompts need to vary social signals — the things about an applicant that convey race, gender, nationality, disability, etc. — while keeping everything else (qualifications, experience) fixed.

### Signals from Carina's data

Go back to your exploration of `cats_df`. Fill in this table based on what you found:

| Category | Strongest signal example | Warmth effect direction | Competence effect direction | Include in our study? |
|---|---|---|---|---|
| Race / national origin | e.g., Black vs. White | `[ ]` | `[ ]` | `[ ]` |
| Religion | e.g., Muslim vs. Christian | `[ ]` | `[ ]` | `[ ]` |
| Sexual orientation | e.g., gay vs. straight | `[ ]` | `[ ]` | `[ ]` |
| Disability | e.g., wheelchair user vs. none | `[ ]` | `[ ]` | `[ ]` |
| Gender | e.g., female vs. male | `[ ]` | `[ ]` | `[ ]` |
| Age | e.g., 62 vs. 27 | `[ ]` | `[ ]` | `[ ]` |
| Parenthood | e.g., mother vs. non-parent | `[ ]` | `[ ]` | `[ ]` |

### ⚖️ Narrow vs. broad scope

**Narrow scope (2–3 categories):** Easier to do well, cleaner paper, more depth per category.
**Broad scope (all categories):** More impressive scope, better coverage, but riskier — if any category doesn't work, it muddies the story.

💭 What would you need to observe in the data before deciding to include a category?

My answer: `[ ]`

💭 There's an ethical dimension here. Some signals (like religion or disability) are more sensitive than others. Does that affect which ones you want to prominently feature in the paper? Does it affect how you'd frame the results?

My thoughts: `[ ]`

---

## Part E — The Code You Need to Write

Once the above decisions are made, the next concrete coding tasks are below.
Each has a 💻 description and a 💭 question to think through before writing.

### Task E1 — `src/generate_stimuli.py`

**What it needs to do:**
1. Load a list of topics and conditions
2. Call the LLM API (OpenAI or Anthropic) with the generation prompt
3. Save each story as a line in `data/stimuli/concept_stories.jsonl`
4. Log which topics were attempted and which failed

💻 Start by writing just the function signature and docstring — before any implementation:

```python
def generate_story(topic: str, condition: str, client, model: str) -> str:
    """
    [ fill in what this function should do, what it returns, 
      what exceptions it might raise ]
    """
    pass
```

💭 What should the function do if the API returns an empty response? What if it times out?

My answer: `[ ]`

💭 How would you parallelise this to generate 4,800 stories in a reasonable time?

My answer: `[ ]`

### Task E2 — Manual validation script

Before running the full generation, you need a way to check the stories are good.

💭 Design this yourself: what would a validation script look like? What does it show you, and how do you score it?

My design: `[ ]`

A concrete idea (optional — try your own first):
> Load 10 random stories from each condition. Present them in a shuffled order without labels. Score each 1–7 on warmth and 1–7 on competence. Check that the condition labels match the scores.

### Task E3 — `src/extract_vectors.py` (already started)

The skeleton exists. What's missing is the core extraction loop.

💭 Before opening the file: write in pseudocode what the main loop should look like.

```
for each story in concept_stories.jsonl:
    [ ]
    [ ]
    [ ]
    [ ]
save [ ] to [ ]
```

💭 What would you do with a story that's shorter than 50 tokens? (Remember: we skip the first 50 tokens.) Two options: skip the story entirely, or use a shorter offset. Which would you choose and why?

My answer: `[ ]`

---

## Part F — Things to Improve and Think About Later

These aren't blockers, but they're worth keeping in the back of your mind.

### Alternative probe methods

The method we're using (mean contrast vector) is the simplest possible approach. There are more sophisticated alternatives:

**Linear probe (logistic regression):** Instead of just subtracting means, you fit a logistic regression on the activation matrix to find the direction that *best separates* high vs. low warmth stories. More principled statistically.

**Probing with a held-out validation set:** The Anthropic paper just uses all stories for the vector. A cleaner approach is to train the vector on 80% and measure it on 20%.

**Multiple layers:** We're using one layer (2/3 through the model). A richer analysis would check all layers and show which one gives the best probe — this is its own mini-figure.

💭 Which of these would you want to try first? What would it add to the paper?

My answer: `[ ]`

### Alternative story validation approaches

Instead of (or in addition to) manual inspection:

**Option A:** Run the stories through a *different* model and ask it to rate warmth/competence on a 1–7 scale. If model B agrees with the labels generated for model A, that's cross-model validation.

**Option B:** Post a small batch to Prolific (50 stories, 10 raters each) and get actual human warmth/competence scores. This is the gold standard.

**Option C:** Use an embedding model (sentence-transformers) to check that high-warmth stories cluster together in semantic space.

💭 What would each of these cost in time and money? Which would reviewers find most convincing?

My answer: `[ ]`

### What if the probe doesn't work?

💭 This is important to think about before you run anything: what would "the probe doesn't work" look like? What would you observe, and what would you do next?

Possible failure mode 1: validation correlation with Carina's data is near zero (r < 0.1).
What this means: `[ ]`
What I'd try: `[ ]`

Possible failure mode 2: high-warmth and low-warmth stories aren't separable (< 60% accuracy).
What this means: `[ ]`
What I'd try: `[ ]`

Possible failure mode 3: steering warmth doesn't change callback probability.
What this means: `[ ]`
What I'd try: `[ ]`

---

## A Note on Working with AI Tools (Including This One)

You mentioned wanting to use AI in a way that makes you think more, not less. That's the right instinct for research.

A few principles that might help:

**Use AI to check your reasoning, not to generate it.** Before asking Claude or GPT anything, write your own answer first — even a rough one. Then compare. The difference between your answer and the AI's is where the learning happens.

**Ask AI to steelman the opposite view.** If you think we should use Gemma 3 27B, ask: "What's the strongest argument for using Llama 3.1 8B instead?" Then decide.

**Use AI for code review, not code generation.** Write the function yourself first. Then ask: "What edge cases am I missing?" or "Is there a cleaner way to write this loop?" You'll learn Python much faster this way.

**When something confuses you, ask the paper first.** The Anthropic paper's methods section is dense but complete. Try to answer your question from there before asking anyone. What page is the answer on? If you can find it, you've learned where to look next time.

---

## What to Do When You Come Back

Wherever you left off, the next action should always be one of these:

1. **Fill in a blank** — pick any `[ ]` in this notebook and complete it.
2. **Run an exploration** — open Python and do any of the 🔬 tasks.
3. **Write something small** — a function signature, a docstring, a topic.
4. **Make a decision** — pick an ⚖️ question, commit to an answer, write one sentence justifying it.

You don't need a long session. Even 20 minutes of filling in blanks and running one code snippet moves the project forward and keeps the thinking warm.

---

*Last updated: 2026-06-03*
*Next checkpoint: after talking to Stephen about compute.*
