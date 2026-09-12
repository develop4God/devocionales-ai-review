"""
Real live ping: makes an actual API call to a provider and prints the response.

Not run as part of the normal test suite -- costs quota/tokens against a real
key. Run explicitly with:

    pytest tests/test_provider_live_ping.py --live -s

or for a specific provider:

    pytest tests/test_provider_live_ping.py --live -s --provider-id groq_gpt_oss_120b
"""

import pytest

from content_batch_graph.domain.providers import get_model


@pytest.fixture
def provider_id(request):
    return request.config.getoption("--provider-id")


def test_provider_live_ping(request, provider_id):
    if not request.config.getoption("--live"):
        pytest.skip("pass --live to actually call the provider")

    model = get_model(provider_id)
    response = model.invoke("Reply with exactly one word: pong")

    print(f"\nprovider: {provider_id or '(default)'}")
    print(f"model class: {type(model).__name__}")
    print(f"response: {response.content!r}")

    assert response.content
