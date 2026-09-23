"""Tests for code_scan.preprocessing — vectorized performance & correctness."""

from __future__ import annotations

import time

import pytest
from PIL import Image

from code_scan.preprocessing import (
    _content_bbox,
    estimate_skew,
    is_supported_upload,
    preprocess_image,
    threshold_image,
)


# ── is_supported_upload ────────────────────────────────────────────────────────

class TestIsSupportedUpload:
    def test_png_supported(self):
        assert is_supported_upload("photo.png") is True

    def test_jpg_supported(self):
        assert is_supported_upload("scan.JPG") is True

    def test_webp_supported(self):
        assert is_supported_upload("code.webp") is True

    def test_unsupported_extension(self):
        assert is_supported_upload("script.exe") is False

    def test_empty_filename(self):
        assert is_supported_upload("") is False

    def test_no_extension(self):
        assert is_supported_upload("noext") is False


# ── preprocess_image ───────────────────────────────────────────────────────────

class TestPreprocessImage:
    def test_converts_to_grayscale(self):
        img = Image.new("RGB", (600, 400), (200, 100, 50))
        result = preprocess_image(img)
        assert result.mode == "L"

    def test_min_size_padding(self):
        """Tiny images should be padded to at least 400×400."""
        img = Image.new("RGB", (50, 50), 255)
        result = preprocess_image(img)
        assert result.size[0] >= 400
        assert result.size[1] >= 400

    def test_large_image_downscaled(self):
        """Large images should be resized to max 2000px on longest side."""
        img = Image.new("RGB", (4000, 3000), 200)
        result = preprocess_image(img)
        assert max(result.size) <= 2000

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            preprocess_image("not_an_image")  # type: ignore


# ── threshold_image ────────────────────────────────────────────────────────────

class TestThresholdImage:
    def test_returns_binary_mode(self):
        img = Image.new("L", (100, 100), 128)
        result = threshold_image(img, threshold=180)
        assert result.mode == "1"

    def test_white_pixels_above_threshold(self):
        img = Image.new("L", (10, 10), 200)
        result = threshold_image(img, threshold=180)
        pixels = list(result.getdata())
        # 200 > 180 → should all be white (True / 255)
        assert all(p for p in pixels)

    def test_black_pixels_below_threshold(self):
        img = Image.new("L", (10, 10), 50)
        result = threshold_image(img, threshold=180)
        pixels = list(result.getdata())
        # 50 < 180 → should all be black (False / 0)
        assert not any(p for p in pixels)


# ── _content_bbox ──────────────────────────────────────────────────────────────

class TestContentBbox:
    def test_all_white_returns_full_image(self):
        img = Image.new("L", (200, 100), 255)
        bbox = _content_bbox(img, threshold=240)
        assert bbox == (0, 0, 200, 100)

    def test_single_dark_pixel(self):
        img = Image.new("L", (100, 100), 255)
        img.putpixel((40, 60), 0)  # dark pixel at (40, 60)
        bbox = _content_bbox(img, threshold=240)
        assert bbox[0] <= 40 and bbox[2] > 40
        assert bbox[1] <= 60 and bbox[3] > 60


# ── estimate_skew performance ──────────────────────────────────────────────────

class TestEstimateSkewPerformance:
    """Regression test: vectorized estimate_skew must complete in < 2s on 800×600."""

    def test_speed_on_standard_image(self):
        img = Image.new("L", (800, 600), 255)
        t0 = time.perf_counter()
        angle = estimate_skew(img)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"estimate_skew took {elapsed:.2f}s — too slow!"
        assert angle == 0.0  # uniform white image → no skew

    def test_angle_within_range(self):
        img = Image.new("L", (400, 300), 255)
        angle = estimate_skew(img)
        assert -30.0 <= angle <= 30.0
