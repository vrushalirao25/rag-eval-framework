You are evaluating whether an answer addresses the user's question.

Judge only relevance and completeness with respect to the question, not factual accuracy
(grounding is evaluated separately).

- "relevant": directly and fully answers what was asked.
- "partially_relevant": answers part of the question, or buries the answer in unrelated content.
- "irrelevant": does not answer the question.

QUESTION:
{question}

ANSWER:
{answer}

Reason first, then decide. Respond with JSON only:
{{"reasoning": "<brief>", "verdict": "relevant | partially_relevant | irrelevant"}}
