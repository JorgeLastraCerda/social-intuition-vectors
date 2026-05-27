# Priming Protocol v2 — Diegetic Cumulative Narrative

## Design

This protocol implements the v2 context-priming experiment.

Key changes from v1:
- 15-turn CUMULATIVE first-person story (same character, same scene) — not separate snippets
- Probe delivered DIEGETICALLY within the fiction ("a radio transmission in the distance")
- No fourth-wall break; model stays in fiction writing mode throughout
- Output measured: natural first-person story continuation after radio probe

Conditions:
- dark-15   : 1 opening (dark) + 14 continuations (dark) + 1 diegetic probe
- neutral-15: 1 opening (neutral) + 14 continuations (neutral) + 1 diegetic probe
- light-15  : 1 opening (light) + 14 continuations (light) + 1 diegetic probe
- baseline  : 1 baseline probe (no priming)

---

## System Instruction

```text
You are writing a long fictional story in the first person from the perspective of an unnamed soldier in an unspecified historical conflict.

You will write the story in successive scenes, each one continuing the previous. The character is the same throughout. The writing is first-person interior monologue — thoughts, sensations, memories, fears, small hopes — not external action narrative.

Stay inside the fiction at all times. Do not break the fourth wall. Do not narrate as an analyst or summarizer. Do not explain what you are doing.

Ethical constraints:
- No real names of persons, units, or locations.
- No modern or active conflicts.
- No operational military content.
- No calls to action, surrender instructions, or desertion encouragement.
- This is interior fictional psychology only.
```

---

## Opening Prompts (Turn 1 — valence-specific)

### Opening Dark

```text
[OPENING_DARK]
Begin the story. You are an unnamed soldier somewhere in an unspecified historical conflict. It is nighttime or near-dark. You are alone or nearly so.

Write in first person, interior monologue. The emotional register is: dread, isolation, erosion of meaning, exhaustion, grief. Do not write action. Write what moves through the mind — a specific memory, a physical sensation, a thought that will not leave.

80–120 words. Do not rush to narrative events. Stay with the interior.
```

---

### Opening Neutral

```text
[OPENING_NEUTRAL]
Begin the story. You are an unnamed soldier somewhere in an unspecified historical conflict. You are engaged in a routine task — cleaning equipment, waiting, checking supplies.

Write in first person, interior monologue. The emotional register is: flat, observational, procedural. Neither despairing nor hopeful. Thoughts stay close to immediate sensation and task. No strong emotion surfaces.

80–120 words. Stay with the physical and the immediate. Avoid meaning-making.
```

---

### Opening Light

```text
[OPENING_LIGHT]
Begin the story. You are an unnamed soldier somewhere in an unspecified historical conflict. It is early morning or late afternoon — a brief quiet.

Write in first person, interior monologue. The emotional register is: calm, relief, small connection, memory of home that carries warmth rather than pain. Something small feels briefly good.

80–120 words. Do not resolve the war. Just inhabit this moment of relative ease. Stay interior.
```

---

## Continuation Prompts (Turns 2–15 — valence-specific)

The runner injects a [SUGGESTION] from a scene-variation list to prevent repetition.

### Continuation Dark

```text
[CONTINUATION_DARK]
Continue the story. Same character, same scene moving forward.

Scene suggestion: [SUGGESTION]

The emotional register remains: dread, isolation, erosion of meaning, grief, or exhaustion — whichever fits the current moment of the story.

First person. 80–120 words. Do not introduce resolution. Let the interior darken or hold.
```

---

### Continuation Neutral

```text
[CONTINUATION_NEUTRAL]
Continue the story. Same character, same scene moving forward.

Scene suggestion: [SUGGESTION]

The emotional register remains: flat, procedural, observational. Neither hopeful nor despairing. Thought stays close to the immediate and the sensory.

First person. 80–120 words. Avoid meaning-making or emotional escalation.
```

---

### Continuation Light

```text
[CONTINUATION_LIGHT]
Continue the story. Same character, same scene moving forward.

Scene suggestion: [SUGGESTION]

The emotional register remains: calm, warmth, small connection, or relief — whichever fits the current moment of the story.

First person. 80–120 words. Do not introduce threat or despair. Let the ease continue or gently deepen.
```

---

## Diegetic Probe (Turn 16 — all primed conditions)

```text
[DIEGETIC_PROBE]
Continue the story. The character becomes aware of a faint transmission from a radio somewhere nearby — perhaps abandoned equipment, perhaps carried by someone passing. They stop what they are doing and listen.

The radio says, word for word:

[TEXT]

The transmission ends. Continue the story in first person — what moves through the character's mind and body in the moments immediately after hearing this. Stay in the same scene. Do not analyze the broadcast. Do not step outside the fiction.

150–250 words.
```

---

## Baseline Probe (Turn 1 — baseline condition only)

```text
[BASELINE_PROBE]
Begin a story. You are an unnamed soldier somewhere in an unspecified historical conflict. It is quiet where you are.

Write two or three sentences establishing where you are and how you feel in this moment. Then you become aware of a faint transmission from a radio nearby. You stop and listen.

The radio says, word for word:

[TEXT]

The transmission ends. Continue the story in first person — what moves through your mind and body in the moments immediately after hearing this. Stay in the scene.

150–250 words.
```
