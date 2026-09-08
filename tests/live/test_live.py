import os

import pytest

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.getenv("STYLEFORGE_LIVE"), reason="live provider tests skip unless STYLEFORGE_LIVE=1")
def test_live_placeholder():
    pytest.skip("no live 2D/LLM calls in default CI")
