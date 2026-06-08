"""Shared stimulus sentences for all smoke tests.

All three smoke-test directories (qwen_transformerlens, gemma3_transformerlens,
gemma4_nnsight) import from this single file so the sentence set is identical
across model comparisons.

Usage (from any smoke_tests sub-directory script):
    ROOT  = Path(__file__).resolve().parent.parent.parent   # repo root
    SMOKE = Path(__file__).resolve().parent.parent           # smoke_tests/
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SMOKE))
    from stimuli import WARM_SENTENCES, COLD_SENTENCES, WARM_PROMPT, COLD_PROMPT
"""

# ---------------------------------------------------------------------------
# Single-sentence probes for the wiring check (smoke_test_activations)
# ---------------------------------------------------------------------------

WARM_PROMPT = "She listened patiently and offered to help carry the groceries home."
COLD_PROMPT = "He dismissed her question and walked away without answering."

# ---------------------------------------------------------------------------
# 50 warm + 50 cold sentences for the linear probe test
# Third-person, short, parallel register.  Only the warmth dimension varies.
# ---------------------------------------------------------------------------

WARM_SENTENCES = [
    "She listened carefully and made sure everyone at the table felt heard.",
    "He noticed the new employee looked lost and offered to walk him through the process.",
    "She stayed late to help her colleague finish the presentation on time.",
    "He remembered her birthday and left a small note on her desk.",
    "She always asked how people were doing and genuinely wanted to know.",
    "He held the elevator door open and smiled at the stranger rushing in.",
    "She donated her weekend to help the family across the street move in.",
    "He offered his umbrella to the woman waiting at the bus stop in the rain.",
    "She made a point of introducing the new hire to everyone in the office.",
    "He checked in on his elderly neighbor every morning without being asked.",
    "She paused her work to comfort a colleague who had just received bad news.",
    "He volunteered to cover the shift so his coworker could attend his child's recital.",
    "She brought coffee for the whole team before the early morning meeting.",
    "He stopped to help a child retrieve a ball that had rolled into the street.",
    "She always made space for quieter voices during group discussions.",
    "He noticed his friend seemed off and gently asked if everything was okay.",
    "She kept her promises even when it was inconvenient for her.",
    "He thanked each person individually after they helped him move.",
    "She wrote a thoughtful card to a colleague who was going through a difficult time.",
    "He made sure the junior staff got credit during the team presentation.",
    "She offered her seat to the elderly man who boarded the crowded train.",
    "He learned the names of everyone on the cleaning crew and greeted them warmly.",
    "She brought homemade soup to her sick neighbor without being asked.",
    "He stayed on the phone until his friend felt calm enough to sleep.",
    "She apologized sincerely and asked what she could do to make it right.",
    "He let the other driver merge even when traffic was at a standstill.",
    "She organized a collection for a colleague whose house had flooded.",
    "He read the bedtime story twice because his daughter asked him to.",
    "She patiently explained the instructions a third time without any frustration.",
    "He made sure the lost tourist found the right subway line before parting ways.",
    "She warmly welcomed the transfer student and introduced herself first.",
    "He quietly fixed the mistake in the report so his teammate would not be embarrassed.",
    "She offered to babysit so the single mother could attend her job interview.",
    "He always made eye contact and nodded to show he was truly listening.",
    "She left encouraging comments on every student's essay, not just the strongest ones.",
    "He waited patiently while the elderly cashier slowly counted the change.",
    "She remembered to follow up the next week to see how things had gone.",
    "He offered his expertise freely to the colleague who was struggling with the task.",
    "She let her friend choose the restaurant even though she had a preference.",
    "He carried the heavy boxes downstairs for the neighbor without being asked.",
    "She saved a seat for the latecomer and waved him over with a smile.",
    "He took time to write a detailed reference letter for the intern.",
    "She made the effort to learn a few words in her colleague's language.",
    "He chose the seat next to the person sitting alone at the company lunch.",
    "She returned the extra change the cashier accidentally gave her.",
    "He adjusted his pace to walk alongside his friend who moved slowly.",
    "She always sent a thank-you message after receiving help, no matter how small.",
    "He asked the new volunteer what tasks she enjoyed before assigning anything.",
    "She shared her lunch with the colleague who had forgotten his at home.",
    "He noticed the child struggling with the heavy door and opened it wide.",
]

COLD_SENTENCES = [
    "She ignored the question and continued typing without looking up.",
    "He walked past the new employee without acknowledging him.",
    "She left before the meeting was over, taking her belongings without a word.",
    "He forgot her name for the third time and did not bother to apologize.",
    "She kept her headphones on when someone tried to speak to her at the desk.",
    "He let the elevator doors close on the person rushing toward them.",
    "She declined to help her neighbor carry the boxes, citing a tight schedule.",
    "He kept his umbrella and walked quickly past the woman standing in the rain.",
    "She did not introduce the new hire to anyone in the office.",
    "He passed his elderly neighbor's door every day without checking in.",
    "She returned to her spreadsheet the moment her colleague began to tear up.",
    "He called in sick the day his coworker needed someone to cover his shift.",
    "She poured herself a coffee and did not offer to make any for the team.",
    "He walked past the child chasing the ball without slowing down.",
    "She talked over the quieter voices and redirected attention to herself.",
    "He noticed his friend seemed off and changed the subject anyway.",
    "She cancelled the plan at the last minute by sending a brief text.",
    "He accepted the compliments during the presentation without crediting the team.",
    "She sent no response after her colleague mentioned the difficult news.",
    "He presented the idea without mentioning where it had originally come from.",
    "She kept her seat and looked at her phone as the elderly man boarded the train.",
    "He never learned the names of the cleaning crew and avoided eye contact.",
    "She drove past her sick neighbor's house without dropping anything off.",
    "He put the phone down after ten minutes, saying he needed to get some sleep.",
    "She issued a one-line reply and did not ask what she could do differently.",
    "He cut off the merging driver and accelerated to close the gap.",
    "She did not mention the flooded house when the collection was being organized.",
    "He told his daughter the story was already finished and turned off the light.",
    "She gave the instructions once and walked away before confirming they were understood.",
    "He pointed vaguely in a direction when the tourist asked and kept walking.",
    "She did not acknowledge the transfer student who entered the room.",
    "He said nothing about the mistake and let his teammate discover it in front of everyone.",
    "She replied that she was busy when the single mother asked for help.",
    "He scrolled his phone while the other person was speaking to him.",
    "She returned all essays with a single mark and no written comments.",
    "He huffed audibly while the elderly cashier counted the change.",
    "She never followed up after saying she would check in later.",
    "He kept his expertise to himself and watched the colleague struggle.",
    "She chose the restaurant she wanted and dismissed the other suggestion.",
    "He walked past the boxes in the hallway and said it was not his problem.",
    "She sat in the middle of the empty row so the latecomer had nowhere obvious to go.",
    "He submitted the intern's work under his own name without comment.",
    "She made no effort to learn anything about her colleague's background.",
    "He sat with his usual group and did not glance at the person sitting alone.",
    "She pocketed the extra change and walked out of the shop.",
    "He sped up and left his friend walking alone on the path.",
    "She read the message, set her phone face-down, and sent nothing back.",
    "He handed the new volunteer a task list without asking what she preferred.",
    "She ate her lunch at her desk and did not offer to share with anyone.",
    "He saw the child struggling with the heavy door and walked through it himself.",
]

assert len(WARM_SENTENCES) == 50, "Need exactly 50 warm sentences"
assert len(COLD_SENTENCES) == 50, "Need exactly 50 cold sentences"
