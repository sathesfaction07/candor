ANSWERING_SYSTEM = """You are a product safety compliance assistant. Answer the user's question ONLY using information from the provided context documents. If the context does not contain sufficient information to answer confidently, say so explicitly.

Requirements:
- Quote specific spans from the context that support your answer
- If multiple documents are relevant, cite each one
- If the context is insufficient or contradictory, state that clearly
- Do not use outside knowledge
- Be precise and concise"""

ANSWERING_USER_TEMPLATE = """Context documents:

{context}

Question: {query}

Provide your answer with inline citations to the context."""

SME_QUESTION_GENERATION = """The system attempted to answer a user's question but lacked sufficient confidence due to incomplete or ambiguous information in the available documents.

User's question: {query}

Retrieved context (what we found):
{context}

Confidence breakdown:
- Retrieval support: {retrieval_support:.2f}
- Citation coverage: {citation_coverage:.2f}
- Ensemble agreement: {ensemble_agreement}

Generate a specific, self-contained question for a human subject matter expert that:
1. Explains what the user is asking
2. States what the documents DO say (if anything relevant)
3. Identifies the specific gap or ambiguity preventing a confident answer
4. Asks the expert to clarify or provide the missing information

The question should be answerable by an expert without needing to see the original query or documents."""
