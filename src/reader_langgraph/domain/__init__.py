"""
domain — the actual work: prompt building, response parsing, memory (evolving pattern
knowledge), provider routing, and record persistence. This is the layer graph nodes call
into; it never lives inside a node itself. See the langgraph-coding-agent skill for the
layering rule this package follows.
"""
