from __future__ import annotations

import json
import random

random.seed(42)


def load_slide_embed(path):
    with open(path, "r") as f:
        return json.load(f)


def top3_correct(case: dict, label: str) -> bool:
    return any(
        e
        for e in (case["prob_top0"], case["prob_top1"], case["prob_top2"])
        if ("0.00" not in e and label in e)
    )


def slide_sample(cases: list[dict], sample_size=0.25):
    correct_cases = [case for case in cases if top3_correct(case, case["label"])]
    incorrect_cases = [case for case in cases if not top3_correct(case, case["label"])]
    for case in correct_cases:
        case["top3_correct"] = True
    for case in incorrect_cases:
        case["top3_correct"] = False
    print(f"correct: {len(correct_cases)} {len(correct_cases) / len(cases):.2%}")
    print(f"incorrect: {len(incorrect_cases)} {len(incorrect_cases) / len(cases):.2%}")
    correct_samples = random.sample(
        correct_cases,
        round(len(correct_cases) * sample_size),
    )
    incorrect_samples = random.sample(
        incorrect_cases,
        round(len(incorrect_cases) * sample_size),
    )
    for case in correct_samples:
        case["sampled"] = True
    for case in incorrect_samples:
        case["sampled"] = True
    assert len([case for case in cases if case.get("sampled", False)]) == round(
        len(correct_cases) * sample_size,
    ) + round(len(incorrect_cases) * sample_size)
    return cases


def main():
    cases = load_slide_embed("slide_vectors.json")
    new_cases = slide_sample(cases)
    with open("slide_vectors_sample.json", "w") as f:
        json.dump(new_cases, f)


if __name__ == "__main__":
    main()
