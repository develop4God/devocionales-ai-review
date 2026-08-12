"""
nodes — one file per graph step. Each node reads state, calls domain logic, returns a
partial state update. No node reimplements domain logic inline. See the
langgraph-coding-agent skill for the layering rule this package follows.
"""
