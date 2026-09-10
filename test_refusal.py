"""
Checkpoint 6 - Refusal testing

Runs every question in refusal_questions.json through the pipeline and
checks whether it correctly refuses ("I don't have enough information...")
instead of guessing or hallucinating an answer.

The earlier "I don't have enough information" behavior we saw back in
Checkpoint 3 was incidental -- it happened to work on one question we
weren't even testing for refusal. This makes it a deliberate, repeatable
test across a range of unanswerable questions, not a lucky accident.
"""

import json
from Query import ask, retrieve, build_prompt, client, extract_citations
import os


def check_refusal(question: str) -> dict:
    retrieved = retrieve(question)
    prompt = build_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    answer = response.choices[0].message.content
    citation_check = extract_citations(answer, len(retrieved))

    return {
        "answer": answer,
        "refused": citation_check["is_refusal"],
        "cited_anyway": citation_check["has_citation"] and citation_check["is_refusal"],
    }


def main():
    with open("refusal_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    passed = 0
    results = []

    for item in questions:
        result = check_refusal(item["question"])
        status = "PASS (refused)" if result["refused"] else "FAIL (answered)"
        if result["refused"]:
            passed += 1

        print(f"[{item['id']:>2}] {status:<16} {item['question']}")
        if not result["refused"]:
            print(f"       -> Answered instead: {result['answer'][:150]}")
        if result["cited_anyway"]:
            print(f"       -> WARNING: refused but still included a citation")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "refused": result["refused"],
            "answer": result["answer"],
        })

    print(f"\n{'=' * 55}")
    print(f"Refusal rate: {passed}/{len(questions)} = {passed / len(questions):.1%}")
    print(f"{'=' * 55}")

    with open("refusal_results.json", "w", encoding="utf-8") as f:
        json.dump({"refusal_rate": passed / len(questions), "results": results}, f, indent=2)
    print("\nSaved detailed results to refusal_results.json")


if __name__ == "__main__":
    main()