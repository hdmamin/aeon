import time

from aeon.utils import timer


def test_timer_measures_duration():
    with timer() as result:
        time.sleep(0.01)

    assert "start" in result
    assert "duration" in result
    assert round(result["duration"], 2) == 0.01
