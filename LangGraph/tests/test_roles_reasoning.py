from content_batch_graph.domain.roles import build_finding_schema, get_role


def test_reasoning_field_is_required_and_first_only_for_opted_in_role():
    direct = build_finding_schema(get_role("native_reader_batch_direct"))
    assert list(direct.model_fields) == ["reasoning", "findings"]
    assert direct.model_fields["reasoning"].is_required()


def test_existing_roles_keep_original_schema():
    for rid in ("native_reader_batch", "native_reader_batch_pt", "native_reader"):
        assert list(build_finding_schema(get_role(rid)).model_fields) == ["findings"]
