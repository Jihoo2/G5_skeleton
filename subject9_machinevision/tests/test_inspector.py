from pathlib import Path
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.cap_inspector import inspect_roi, judge_cap, load_rule, measure_area_ratio
from src.fastapi_cap_app import app, state


def synthetic_roi(cap_on: bool) -> np.ndarray:
    """실제 사진 없이 검사 규칙을 확인하는 인공 패턴."""
    image = np.full((160, 200, 3), 235, np.uint8)
    if cap_on:
        cv2.rectangle(image, (35, 25), (165, 105), (15, 15, 15), -1)
    else:
        cv2.rectangle(image, (65, 90), (135, 120), (15, 15, 15), -1)
    return image


def test_rule(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "cap_on.jpg"), synthetic_roi(True))
    cv2.imwrite(str(tmp_path / "cap_off.jpg"), synthetic_roi(False))
    rule = load_rule(tmp_path)

    on = inspect_roi(synthetic_roi(True), rule)
    off = inspect_roi(synthetic_roi(False), rule)

    assert on["decision"] == "OK"
    assert off["decision"] == "NG"
    assert rule.cap_on_ratio > rule.threshold > rule.cap_off_ratio
    assert measure_area_ratio(synthetic_roi(True))[0] > measure_area_ratio(synthetic_roi(False))[0]
    assert judge_cap(rule.threshold, rule)[0] == "OK"


def test_api_and_fake_camera(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "cap_on.jpg"), synthetic_roi(True))
    cv2.imwrite(str(tmp_path / "cap_off.jpg"), synthetic_roi(False))
    fake_rule = load_rule(tmp_path)

    import src.fastapi_cap_app as server
    original = server.cv2.VideoCapture
    which = {"on": True}

    class FakeCapture:
        def __init__(self, url):
            self.url = url
        def isOpened(self):
            return True
        def read(self):
            time.sleep(0.04)
            return True, synthetic_roi(which["on"])
        def release(self):
            pass

    server.cv2.VideoCapture = FakeCapture
    try:
        with TestClient(app) as client:
            with state.lock:
                state.rule = fake_rule
                state.rule_error = ""

            assert client.get("/").status_code == 200
            assert client.post("/api/camera", json={"url": "https://example.com/video"}).status_code == 400
            assert client.post("/api/camera", json={"url": "http://192.168.0.30:8080/video"}).status_code == 200

            assert client.post("/api/roi", json={"x1": 0.4, "y1": 0.3, "x2": 0.3, "y2": 0.4}).status_code == 400
            assert client.post("/api/roi", json={"x1": 0, "y1": 0, "x2": 1, "y2": 1}).status_code == 200

            def wait_for(d):
                status = {}
                for _ in range(60):
                    status = client.get("/api/status").json()
                    if status["decision"] == d and status["camera_status"] == "ONLINE":
                        return status
                    time.sleep(0.05)
                raise AssertionError(f"No status {d}: {status}")

            assert wait_for("OK")["ratio"] > fake_rule.threshold
            which["on"] = False
            assert wait_for("NG")["ratio"] < fake_rule.threshold

            assert client.delete("/api/roi").json()["ok"]
    finally:
        server.cv2.VideoCapture = original
        state.set_camera("")