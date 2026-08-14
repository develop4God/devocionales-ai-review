import json
from datetime import date

from batch_common import BatchProviderConfig

from content_batch_graph.domain.devotional_gen import (
    build_generation_system,
    build_generation_user,
    build_year_batch,
    custom_id_for,
    language_name,
    year_specs,
)
from content_batch_graph.domain.roles import get_role

PROVIDER = BatchProviderConfig(
    provider_id="p_batch",
    base_url="https://x.test/v1",
    model="accounts/x/models/m1",
    env_var="X_API_KEY",
)


def _role():
    return get_role("devotional_author")


# ── year_specs ────────────────────────────────────────────────────────────────


def test_year_specs_non_leap_year_has_365_days():
    days = year_specs(2026)
    assert len(days) == 365
    assert days[0] == date(2026, 1, 1)
    assert days[-1] == date(2026, 12, 31)


def test_year_specs_leap_year_has_366_days_including_feb_29():
    days = year_specs(2028)
    assert len(days) == 366
    assert date(2028, 2, 29) in days
    assert days[-1] == date(2028, 12, 31)


def test_year_specs_is_sorted_and_has_no_duplicates():
    days = year_specs(2026)
    assert days == sorted(days)
    assert len(set(days)) == len(days)


# ── prompts ───────────────────────────────────────────────────────────────────


def test_generation_system_substitutes_language_and_states_json_contract():
    system = build_generation_system("es", "RVR1960", _role())
    assert "{language}" not in system
    # The persona reads "a native {language} speaker" — it must resolve to a name,
    # not the bare ISO code.
    assert "Spanish" in system
    assert "native es speaker" not in system
    assert "RVR1960" in system
    assert "reflexion" in system and "oracion" in system


def test_generation_system_maps_known_codes_to_language_names():
    assert language_name("tl") == "Tagalog"
    assert language_name("FIL") == "Filipino"


def test_generation_system_passes_unknown_language_through_unchanged():
    assert language_name("xx") == "xx"
    assert "xx" in build_generation_system("xx", "V", _role())


def test_generation_system_is_identical_across_days_for_caching():
    a = build_generation_system("es", "RVR1960", _role())
    b = build_generation_system("es", "RVR1960", _role())
    assert a == b


def test_generation_user_carries_the_date_and_nothing_verse_related():
    user = build_generation_user(date(2026, 3, 5))
    assert "2026-03-05" in user
    assert "Thursday" in user
    for forbidden in ("verse", "Bible", "chapter"):
        assert forbidden.lower() not in user.lower()


def test_custom_id_format_is_zero_padded():
    assert (
        custom_id_for("es", "RVR1960", date(2026, 1, 7)) == "gen_es_RVR1960_2026_01-07"
    )


# ── build_year_batch ──────────────────────────────────────────────────────────


def test_build_year_batch_produces_one_record_per_day():
    records = build_year_batch(2026, "es", "RVR1960", PROVIDER, _role())
    assert len(records) == 365
    assert len(build_year_batch(2028, "es", "RVR1960", PROVIDER, _role())) == 366


def test_build_year_batch_custom_ids_are_unique_and_deterministic():
    records = build_year_batch(2026, "tl", "ASND", PROVIDER, _role())
    ids = [r["custom_id"] for r in records]
    assert len(set(ids)) == 365
    assert ids[0] == "gen_tl_ASND_2026_01-01"
    assert ids[-1] == "gen_tl_ASND_2026_12-31"


def test_build_year_batch_record_body_shape():
    record = build_year_batch(2026, "es", "RVR1960", PROVIDER, _role())[0]
    assert record["method"] == "POST"
    assert record["url"] == "/v1/chat/completions"
    body = record["body"]
    assert body["model"] == PROVIDER.model
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0.7
    assert body["response_format"] == {"type": "json_object"}
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]


def test_build_year_batch_shares_one_system_prompt_across_all_records():
    records = build_year_batch(2026, "es", "RVR1960", PROVIDER, _role())
    systems = {r["body"]["messages"][0]["content"] for r in records}
    assert len(systems) == 1


def test_build_year_batch_user_prompts_are_all_distinct():
    records = build_year_batch(2026, "es", "RVR1960", PROVIDER, _role())
    users = {r["body"]["messages"][1]["content"] for r in records}
    assert len(users) == 365


def test_build_year_batch_honors_overrides():
    record = build_year_batch(
        2026, "es", "RVR1960", PROVIDER, _role(), max_tokens=512, temperature=0.2
    )[0]
    assert record["body"]["max_tokens"] == 512
    assert record["body"]["temperature"] == 0.2


def test_build_year_batch_merges_provider_extra_record_fields():
    provider = BatchProviderConfig(
        provider_id="p",
        base_url="https://x.test/v1",
        model="m",
        env_var="K",
        extra_record_fields={"top_p": 0.9},
    )
    record = build_year_batch(2026, "es", "RVR1960", provider, _role())[0]
    assert record["body"]["top_p"] == 0.9


def test_build_year_batch_records_are_json_serializable():
    for record in build_year_batch(2026, "es", "RVR1960", PROVIDER, _role())[:5]:
        assert json.loads(json.dumps(record)) == record
