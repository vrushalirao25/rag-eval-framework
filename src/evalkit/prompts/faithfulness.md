You are evaluating whether an answer is grounded in the provided policy excerpts.

Steps:
1. Break the ANSWER into atomic factual claims (one fact per claim).
2. For each claim, decide whether it is directly supported by the EXCERPTS.
   - "supported": the excerpts state it, or it follows with no added assumptions.
   - "unsupported": the excerpts do not state it, contradict it, or it adds detail not present.
3. Quote the shortest piece of evidence for each supported claim.

Do not use outside knowledge. Do not reward fluency or length. Reason before you decide.

EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Respond with JSON only:
{{"reasoning": "<brief>", "claims": [{{"claim": "<text>", "supported": true, "evidence": "<quote or empty>"}}]}}
