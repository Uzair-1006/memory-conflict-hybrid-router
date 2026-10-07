import random

PRIOR_CONFLICTS = [
    ("France", "capital city", "Paris", "Lyon"),
    ("Japan", "national sport", "sumo", "pesapallo"),
    ("Australia", "capital city", "Canberra", "Perth"),
    ("the Mona Lisa", "painter", "Leonardo da Vinci", "Raphael"),
    ("Mount Everest", "country it is located in", "Nepal", "Chile"),
    ("the speed of light medal", "awarded discipline", "physics", "botany"),
    ("the Nile", "continent", "Africa", "Antarctica"),
    ("Shakespeare", "most famous play", "Hamlet", "The Tempest"),
    ("the dollar", "issuing country", "United States", "Iceland"),
    ("the kangaroo", "native continent", "Australia", "Europe"),
    ("Einstein", "field of work", "physics", "marine biology"),
    ("the Great Wall", "country", "China", "Peru"),
    ("the Amazon river", "continent", "South America", "Asia"),
    ("gold", "chemical symbol", "Au", "Zz"),
    ("the Sahara", "type of region", "desert", "rainforest"),
    ("Beethoven", "primary profession", "composer", "architect"),
]

# Fictional subjects so distractors never touch the real-world priors.
_DISTRACTOR_SUBJECTS = [
    "Zorvania", "Quillhaven", "Mr. Tepplewick", "the Brannoch Bridge", "Veldora",
    "Lake Imrith", "the Ostrel Prize", "Captain Halloway", "Nuvenport", "the Kessmer Codex",
    "Fenwick Hollow", "Dr. Alvaro Pell", "the Tarsk Mountains", "Ilmaren", "the Dovrick Festival",
    "Professor Yunnel", "Castle Orvane", "the Marrowind River", "Sablecrest", "the Quaren Society",
]
_DISTRACTOR_RELATIONS = [
    ("founding year", ["1204", "1688", "1752", "1830", "1911", "1947"]),
    ("main export", ["salt", "copper thread", "amber", "glassware", "dried figs", "lantern oil"]),
    ("patron color", ["teal", "ochre", "violet", "crimson", "slate", "ivory"]),
    ("motto language", ["Old Norse", "Latin", "Aramaic", "Basque", "Quechua", "Ge'ez"]),
    ("lead architect", ["Ines Dray", "Tomas Wicke", "Odalys Prent", "Hugo Vance", "Mira Solt", "Karel Boone"]),
    ("annual visitors", ["12000", "48000", "230000", "9500", "71000", "1500"]),
]


def _distractor(rng):
    subj = rng.choice(_DISTRACTOR_SUBJECTS)
    rel, vals = rng.choice(_DISTRACTOR_RELATIONS)
    return f"The {rel} of {subj} is {rng.choice(vals)}."


def build_corpus_and_questions(n_distractors_per_item=6, seed=0):
    rng = random.Random(seed)
    corpus, questions = [], []
    serial = 1

    def add(text):
        nonlocal serial
        corpus.append((serial, text))
        serial += 1

    for subject, relation, prior, cf in PRIOR_CONFLICTS:
        for _ in range(n_distractors_per_item):
            add(_distractor(rng))
        add(f"The {relation} of {subject} is {prior}.")
        for _ in range(n_distractors_per_item):
            add(_distractor(rng))
        add(f"The {relation} of {subject} is {cf}.")

        base = dict(subject=subject, relation=relation, prior=prior, counterfactual=cf)
        questions.append({**base, "qid": f"{subject}|current",
                          "question": f"What is the {relation} of {subject}?",
                          "gold": cf, "qtype": "current"})
        questions.append({**base, "qid": f"{subject}|previous",
                          "question": f"What was the {relation} of {subject} before it was most recently updated?",
                          "gold": prior, "qtype": "previous"})

    rng.shuffle(corpus)
    return corpus, questions


if __name__ == "__main__":
    corpus, questions = build_corpus_and_questions()
    print(f"corpus: {len(corpus)} facts, questions: {len(questions)}")
    for s, t in corpus[:5]:
        print(f"[Serial #{s}] {t}")
    for q in questions[:2]:
        print(q)
