"""
Manual-review corrections for the ES fine-tuning dataset's true_positive tier.

Each entry below records a judgment made by hand-reviewing all 107 true_positive
records against the actual corpus text (see conversation history for the review).
Applied by build_finetune_dataset_es.py before full-text backfill.
"""

# quoted_text substrings that identify records to DROP entirely from true_positive
# (confirmed wrong/hallucinated fixes -- the proposed_fix is worse than the original)
DROP_QUOTED_SUBSTRINGS = [
    "la honra",              # 1Pedro16-7: "honra" is correct, "honor" is an unneeded register swap
    "Tu honra",              # 2Timoteo222: same pattern
    "Todo lo que hago es para tu honra",  # Hechos2831: same pattern
    "el labrador divino",    # Juan1224-26: "laborador" is not a real word -- fix goes backward
    "un lebrillo",           # Juan1315: "lebrillo" (wash basin) is correct; "pañuelo" is nonsensical here
    "Tu mies",               # Romanos126: "mies" (harvest, biblical) is correct; "Tu mi" is a fragment
    "Jesucristo",            # Filipenses39: "Jesucristo" is the standard correct form
]

# quoted_text substrings for records that are a duplicate of the hard_negative_partial_fix
# entry already captured separately -- remove to avoid contradictory training signal
DROP_DUPLICATE_SUBSTRINGS = [
    "a la verdadera discipulado. Jesús nos presenta",  # buggy interim version, kept only in hard_negative_partial_fix
]

# quoted_text substrings that are real commit changes but NOT typo/grammar errors --
# recategorize instead of dropping, so they don't teach the model that style/content
# preference edits are linguistic errors
RECATEGORIZE = {
    # book-name numeral style ("Primera Corintios" vs "1 Corintios") -- both valid, house style only
    "Primera Juan": "style_preference",
    "Primera Timoteo": "style_preference",
    "Segunda Timoteo 3:16-17": "style_preference",
    "Primera Corintios": "style_preference",
    "Segunda Corintios 1:21-22": "style_preference",
    "Segunda Corintios 4:18": "style_preference",
    "Segunda Corintios 3:5": "style_preference",
    "Primera Corintios 6:11": "style_preference",
    "Primera Corintios 3:6": "style_preference",
    "Primera Corintios 9:24-25": "style_preference",
    "Segunda Corintios 6:4-5": "style_preference",
    "Primera Corintios 15:10": "style_preference",
    "1Corintios1013": "style_preference",  # entry_id fallback match

    # content/theological rewrites bundled into a grammar-labeled commit
    "y que pueda siempre animar y edificar a los": "style_preference",
    "pueda siempre animar y edificar a los demás": "style_preference",
    "obedecerte de corazón, a vivir de acuerdo con tu voluntad y a": "content_rewrite",
    "Este acto de sanidad física es una clara manifestación": "content_rewrite",
    "solo Dios puede hacer. La sanidad física del paralítico": "content_rewrite",
    "nueva y plena. La sanidad física es una manifestación del": "content_rewrite",
    "poder de Dios, pero la sanidad espiritual es la que": "content_rewrite",
    "la sanidad espiritual es la que nos permite experimentar": "content_rewrite",
    "santo, morada de tu Espíritu. Guíanos a ser más": "content_rewrite",
}

# quoted_text substrings needing a manual sentence-context recheck before trusting --
# excluded from the cleaned true_positive tier until verified
HOLD_FOR_RECHECK = [
    "la Vid verdadera",     # Juan157: likely a deliberate Jn 15:1 allusion -- fix may remove intended reference
    "presencia el dolor",   # Juan1135-36: triage report says even the proposed fix may still be wrong
    "el guía",              # Colosenses39: "el guía" vs "la guía" are different words, not a gender typo
    "inescrutable",         # Romanos920-21: both words are valid -- looks like preference, not error
    "a Tu señorío",         # Colosenses26-7: "señorío" is valid theological vocab -- looks like preference
]


def classify(record):
    """Return ('drop', reason) | ('recategorize', new_category) | ('hold', reason) | ('keep', None)."""
    q = record.get("quoted_text", "") or ""
    entry_id = record.get("source", "")

    for s in DROP_QUOTED_SUBSTRINGS:
        if s in q:
            return ("drop", "confirmed_wrong_fix")
    for s in DROP_DUPLICATE_SUBSTRINGS:
        if s in q:
            return ("drop", "duplicate_of_hard_negative_partial_fix")
    for s in HOLD_FOR_RECHECK:
        if s in q:
            return ("hold", "needs_manual_recheck")
    for s, new_cat in RECATEGORIZE.items():
        if s in q or s in entry_id:
            return ("recategorize", new_cat)
    return ("keep", None)
