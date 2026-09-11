"""
One-time manual patch of LangGraph/data/checkpoints/2027_es_apply_ledger.jsonl.

Adds a `finding_category` field to every critic_finding, classifying it into one
of the five categories established by hand-review (see conversation history /
LangGraph/scripts/finetune_dataset_corrections.py for the ES base-file equivalent):

  1. true_positive              -- real error, is_valid=True, correct fix, no drift
  2. true_negative_clean        -- no finding at all (nothing to patch per-finding)
  3. hard_negative_false_positive -- is_valid=False, OR is_valid=True but manually
                                     confirmed wrong after hand review (flipped here)
  4. hard_negative_bad_fix      -- is_valid=True but drift_detected=True on this entry
  5. hard_negative_partial_fix  -- not present in this ledger (found only in the
                                     2025/2026 base-file commit history)

This is a manual, one-time correction -- NOT a pipeline/graph change. If the
fine-tuning approach proves useful, closing this loop inside the graph itself
(a holistic post-fix re-review node) is future work, not done here.

Run: python3 patch_2027_es_ledger_finding_categories.py
Writes the ledger in place; a .bak copy is kept alongside it.
"""
import json
import shutil
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "data" / "checkpoints" / "2027_es_apply_ledger.jsonl"

# Confirmed-wrong findings from manual review: the critic marked these is_valid=True
# and a fix was recorded, but hand review against the live corpus + ground-truth
# Bible DB showed the ORIGINAL text was already correct. Flip to false_positive.
CONFIRMED_WRONG = {
    ("1Pedro16-7RVR196020270803", "la honra"),
    ("2Timoteo222RVR196020271108", "Tu honra"),
    ("Hechos2831RVR196020280329", "Todo lo que hago es para tu honra y para la expansión de tu santo nombre en este mundo."),
    ("Juan1224-26RVR196020271023", "el labrador divino"),
    ("Juan1315RVR196020271027", "un lebrillo"),
    ("Romanos126RVR196020271228", "Tu mies"),
    ("Filipenses39RVR196020280618", "Jesucristo"),
}

# quoted_text substrings needing manual recheck before trusting either way --
# not flipped, just tagged so they're excluded from training until resolved
HOLD_FOR_RECHECK_SUBSTRINGS = [
    "la Vid verdadera",
    "presencia el dolor",
    "el guía",
    "inescrutable",
    "a Tu señorío",
]


# Systematic critic mistakes: any finding proposing one of these exact
# replacements is wrong regardless of surrounding text or entry_id, because the
# "error" it flags is not an error in standard Spanish. Found by noticing the
# same bad pattern recurring at multiple entries beyond the first hand-reviewed case.
CONFIRMED_WRONG_FIX_PATTERNS = {
    ("honra", "honor"),          # "honra" is correct; critic keeps proposing "honor"
    ("Jesucristo", "Jesús Cristo"),  # "Jesucristo" is the standard correct form
    ("labrador", "laborador"),   # "laborador" is not a real word
    ("lebrillo", "pañuelo"),     # basin -> handkerchief makes no sense in a foot-washing context
    ("mies", "mi"),              # "mies" (harvest) is correct; "mi" is a broken fragment
}


def _fix_pattern_match(quoted, replacement):
    if not quoted or not replacement:
        return False
    for wrong_word, bad_replacement in CONFIRMED_WRONG_FIX_PATTERNS:
        if wrong_word in quoted and bad_replacement in replacement:
            return True
    return False


def classify_finding(entry_id, field, finding, entry_drift_detected):
    quoted = finding.get("quoted_text", "") or ""
    replacement = finding.get("replacement_text", "") or ""
    is_valid = finding.get("is_valid")

    if (entry_id, quoted) in CONFIRMED_WRONG or _fix_pattern_match(quoted, replacement):
        return "hard_negative_false_positive", "manually_reviewed_confirmed_wrong"

    for s in HOLD_FOR_RECHECK_SUBSTRINGS:
        if s in quoted:
            return "hold_for_recheck", "manually_reviewed_ambiguous"

    if is_valid is False:
        return "hard_negative_false_positive", "critic_self_dismissed"

    if is_valid is True:
        if entry_drift_detected:
            return "hard_negative_bad_fix", "drift_check_reverted"
        return "true_positive", "critic_verified"

    return "unclassified", "no_is_valid_field"


def main():
    backup = LEDGER.with_suffix(LEDGER.suffix + ".bak-pre-category-patch")
    if not backup.exists():
        shutil.copy(LEDGER, backup)
        print(f"Backup written to {backup}")

    with open(LEDGER, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    counts = {}
    flipped = 0
    for entry in lines:
        drift = bool(entry.get("drift_detected"))
        for finding in entry.get("critic_findings", []):
            category, reason = classify_finding(entry["entry_id"], entry["field"], finding, drift)
            finding["finding_category"] = category
            finding["finding_category_reason"] = reason
            counts[category] = counts.get(category, 0) + 1
            if category == "hard_negative_false_positive" and reason == "manually_reviewed_confirmed_wrong":
                if finding.get("is_valid") is True:
                    finding["is_valid"] = False
                    finding["is_valid_overridden"] = True
                    flipped += 1

    with open(LEDGER, "w", encoding="utf-8") as f:
        for entry in lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("finding_category counts:", counts)
    print("is_valid flipped True->False:", flipped)


if __name__ == "__main__":
    main()
