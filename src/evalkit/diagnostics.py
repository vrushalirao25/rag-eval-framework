"""Maps each evaluation dimension to the system components most likely
responsible when it degrades. This is what turns a score into an engineering
action."""

DIMENSIONS = {
    "schema_validity": {
        "label": "Structure – output contract",
        "method": "code",
        "likely_causes": "Output format instructions, missing structured-output / JSON mode, no validate-and-retry step.",
    },
    "context_precision": {
        "label": "Retrieval – precision",
        "method": "code",
        "likely_causes": "Noisy context: chunk size, missing re-ranker, missing metadata filters, top_k too high.",
    },
    "context_recall": {
        "label": "Retrieval – recall",
        "method": "code",
        "likely_causes": "Evidence not retrieved: chunking boundaries, embedding model, top_k too low, no query rewriting.",
    },
    "faithfulness": {
        "label": "Grounding – faithfulness",
        "method": "llm-judge",
        "likely_causes": "Grounding instructions in the prompt, context construction/ordering, model choice or temperature.",
    },
    "citation_accuracy": {
        "label": "Citations – accuracy",
        "method": "code",
        "likely_causes": "Chunk-ID propagation into the prompt, citation instructions, no post-generation citation validation.",
    },
    "answer_relevance": {
        "label": "Answer – relevance",
        "method": "llm-judge",
        "likely_causes": "Prompt focus, irrelevant context distracting the model, multi-part questions not decomposed.",
    },
    "scope_adherence": {
        "label": "Scope – adherence",
        "method": "llm-judge",
        "likely_causes": "System prompt boundaries, missing guardrails for advice/speculation.",
    },
    "refusal_accuracy": {
        "label": "Scope – refusal accuracy",
        "method": "code",
        "likely_causes": "False refusals → retrieval misses or over-strict instructions. "
        "Missed refusals → no evidence-sufficiency check or retrieval-score threshold.",
    },
}
