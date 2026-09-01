from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SUPPORTED_UPLOAD_FORMATS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".heic",
    ".pdf",
]

ImageLike = Union[str, Path, Image.Image]


def is_supported_upload(filename: str) -> bool:
    """Return True when the file extension matches a supported ingestion format."""
    if not filename:
        return False
    suffix = Path(filename).suffix.lower()
    return suffix in SUPPORTED_UPLOAD_FORMATS


def _normalize_size(image: Image.Image, max_dimension: int = 2000, min_dimension: int = 400) -> Image.Image:
    width, height = image.size
    scale = 1.0

    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
    elif min(width, height) < min_dimension:
        scale = max(scale, min_dimension / min(width, height))

    if scale != 1.0:
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def _content_bbox(image: Image.Image, threshold: int = 240) -> tuple[int, int, int, int]:
    """Return the bounding box of non-white pixels in an image."""
    bbox = []
    width, height = image.size

    for y in range(height):
        for x in range(width):
            if image.getpixel((x, y)) < threshold:
                bbox.append((x, y))

    if not bbox:
        return (0, 0, width, height)

    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Normalize an uploaded image into a clean grayscale form for downstream OCR and layout analysis."""
    if not isinstance(image, Image.Image):
        raise TypeError("preprocess_image expects a PIL.Image.Image instance")

    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = _normalize_size(image)

    enhancer = ImageEnhance.Contrast(image).enhance(1.8)
    image = enhancer.filter(ImageFilter.MedianFilter)

    if image.size[0] < 400 or image.size[1] < 400:
        padded = Image.new("L", (max(400, image.size[0]), max(400, image.size[1])), 255)
        padded.paste(image, ((padded.size[0] - image.size[0]) // 2, (padded.size[1] - image.size[1]) // 2))
        image = padded

    return image


def threshold_image(image: Image.Image, threshold: int = 180) -> Image.Image:
    """Convert a grayscale image to a binary image for layout and skew analysis."""
    processed = image.convert("L")
    return processed.point(lambda p: 0 if p < threshold else 255, mode="1")


def estimate_skew(image: Image.Image) -> float:
    """Estimate the skew angle of a page using row-density variance across candidate rotations."""
    gray = threshold_image(preprocess_image(image), threshold=200)
    if gray.size[0] == 0 or gray.size[1] == 0:
        return 0.0

    best_angle = 0.0
    best_score = -1.0

    for angle in range(-30, 31):
        rotated = gray.rotate(angle, expand=True, fillcolor=255)
        row_counts = []
        for y in range(rotated.height):
            count = 0
            for x in range(rotated.width):
                if rotated.getpixel((x, y)) == 0:
                    count += 1
            row_counts.append(count)

        score = 0
        for index in range(1, len(row_counts)):
            score += abs(row_counts[index] - row_counts[index - 1])

        if score > best_score or (abs(score - best_score) < 1e-9 and abs(angle) < abs(best_angle)):
            best_score = score
            best_angle = float(angle)

    return best_angle


def preprocess_document(image: Image.Image) -> dict[str, Any]:
    """Deskew and normalize a page image while returning a bounding box for the content region."""
    processed = preprocess_image(image)
    angle = estimate_skew(processed)
    deskewed = processed.rotate(-angle, expand=True, fillcolor=255)
    binary = threshold_image(deskewed)
    bounds = _content_bbox(binary, threshold=128)
    cropped = deskewed.crop(bounds)
    return {"processed": cropped, "bounds": bounds, "skew_angle": angle}


def segment_layout(image: Image.Image) -> list[dict[str, Any]]:
    """Divide a page into coarse layout regions for downstream recognition and OCR."""
    processed = preprocess_image(image)
    thresholded = threshold_image(processed)
    width, height = thresholded.size
    dark_pixels = []

    for y in range(height):
        for x in range(width):
            if thresholded.getpixel((x, y)) == 0:
                dark_pixels.append((x, y))

    if not dark_pixels:
        return [
            {"label": "text", "bbox": (0, 0, width, height // 2)},
            {"label": "code", "bbox": (0, height // 2, width, height)},
        ]

    ys = [point[1] for point in dark_pixels]
    split_y = max(0, min(height - 1, sum(ys) // len(ys)))

    return [
        {"label": "text", "bbox": (0, 0, width, split_y)},
        {"label": "code", "bbox": (0, split_y, width, height)},
    ]


def process_upload(file_path: ImageLike) -> dict[str, Any]:
    """Validate, open, and preprocess an uploaded document or image.

    The return value is intentionally lightweight and designed to be extended into a richer
    Phase 1 ingestion metadata payload.
    """
    file_name = str(file_path)
    is_supported = is_supported_upload(file_name)

    if not is_supported:
        return {"is_supported": False, "format": None, "processed_size": None, "source": file_name}

    image = Image.open(file_path)
    processed = preprocess_image(image)

    return {
        "is_supported": True,
        "format": image.format or Path(file_name).suffix.upper().lstrip("."),
        "processed_size": processed.size,
        "source": file_name,
    }
