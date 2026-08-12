"""
content_batch_graph — a LangGraph pipeline automating a batch content review protocol:
translate/generate → independent critic rounds → verified triage → fix → validate →
human confirmation, repeated per phase, with durable patterns proposed (never written)
back for human approval at the end.

A new project, not a port. It reuses concepts proven in two prior manual systems —
GEP_Genome-Evolution-Protocol's reader-persona prompting, evolving pattern memory, and
audited review trail; and a separate manual translate-batch protocol's two-round
independent critic discipline, verify-before-trust triage, and real-validator gates
between LLM judgment steps — but reimplements them here on LangGraph/LangChain with its
own code and its own tests, not by importing either prior system.
"""
