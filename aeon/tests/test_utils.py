import time

from aeon.utils import timer


class TestTimer:

    def test_timer_measures_duration(self):
        with timer() as result:
            time.sleep(0.01)

        assert "duration" in result
        assert result["duration"] >= 0.01
        assert result["duration"] < 0.1  # Should complete quickly

    def test_timer_includes_start_time(self):
        with timer() as result:
            pass

        assert "start" in result
        assert "duration" in result
