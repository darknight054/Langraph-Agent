"""Image extraction from OCR output using reference tags with bounding boxes.

DeepSeek OCR with grounding outputs reference tags that identify images, tables,
and figures in the document along with their bounding box coordinates. This module
extracts those referenced regions and replaces the tags with markdown image links.

Reference format:
    <|ref|>image<|/ref|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

Where coordinates are normalized to 0-999 range.
"""

import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from common import get_logger

log = get_logger(__name__)


@dataclass
class ExtractedImage:
    """Represents an extracted image from OCR output."""

    ref_type: str  # e.g., "image", "figure", "table", "chart"
    page_number: int
    image_index: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in original coords
    image: Image.Image
    file_path: Path | None = None


class ImageExtractor:
    """Extract images from OCR output using reference tags with coordinates.

    This class parses OCR output for reference tags that identify embedded
    images/figures/tables/charts, extracts the corresponding regions from
    the source page image, and replaces the tags with markdown image links.
    """

    # Pattern for reference tags with bounding boxes
    # Matches: <|ref|>TYPE<|/ref|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
    # Also handles full-width unicode variants
    REFERENCE_PATTERN = re.compile(
        r'<[||]ref[||]>(\w+)<[||]/ref[||]>'
        r'<[||]box_start[||]>\((\d+),(\d+)\),\((\d+),(\d+)\)<[||]box_end[||]>',
        re.IGNORECASE
    )

    # Coordinate normalization factor (DeepSeek uses 0-999 range)
    COORD_MAX = 999

    def __init__(self, output_dir: Path | str | None = None):
        """Initialize image extractor.

        Args:
            output_dir: Directory to save extracted images.
                       If None, images are extracted but not saved.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_references(self, text: str) -> list[tuple[str, int, int, int, int, int, int]]:
        """Find all reference tags in text.

        Args:
            text: OCR output text containing reference tags

        Returns:
            List of tuples: (ref_type, x1, y1, x2, y2, match_start, match_end)
        """
        references = []
        for match in self.REFERENCE_PATTERN.finditer(text):
            ref_type = match.group(1).lower()
            x1 = int(match.group(2))
            y1 = int(match.group(3))
            x2 = int(match.group(4))
            y2 = int(match.group(5))
            references.append((ref_type, x1, y1, x2, y2, match.start(), match.end()))
        return references

    def scale_coordinates(
        self,
        x: int,
        y: int,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int]:
        """Scale normalized coordinates (0-999) to image dimensions.

        Args:
            x: X coordinate in 0-999 range
            y: Y coordinate in 0-999 range
            img_width: Actual image width
            img_height: Actual image height

        Returns:
            Tuple of (scaled_x, scaled_y)
        """
        scaled_x = int(x * img_width / self.COORD_MAX)
        scaled_y = int(y * img_height / self.COORD_MAX)
        return scaled_x, scaled_y

    def extract_region(
        self,
        page_image: Image.Image,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Image.Image:
        """Extract a region from the page image.

        Args:
            page_image: Source page image
            x1, y1: Top-left corner (normalized 0-999)
            x2, y2: Bottom-right corner (normalized 0-999)

        Returns:
            Cropped image region
        """
        img_width, img_height = page_image.size

        # Scale coordinates to actual image dimensions
        scaled_x1, scaled_y1 = self.scale_coordinates(x1, y1, img_width, img_height)
        scaled_x2, scaled_y2 = self.scale_coordinates(x2, y2, img_width, img_height)

        # Ensure coordinates are within bounds
        scaled_x1 = max(0, min(scaled_x1, img_width))
        scaled_y1 = max(0, min(scaled_y1, img_height))
        scaled_x2 = max(0, min(scaled_x2, img_width))
        scaled_y2 = max(0, min(scaled_y2, img_height))

        # Ensure x1 < x2 and y1 < y2
        if scaled_x1 > scaled_x2:
            scaled_x1, scaled_x2 = scaled_x2, scaled_x1
        if scaled_y1 > scaled_y2:
            scaled_y1, scaled_y2 = scaled_y2, scaled_y1

        # Crop the region
        return page_image.crop((scaled_x1, scaled_y1, scaled_x2, scaled_y2))

    def generate_filename(
        self,
        document_name: str,
        page_number: int,
        image_index: int,
        ref_type: str,
    ) -> str:
        """Generate a filename for the extracted image.

        Args:
            document_name: Name of the source document
            page_number: Page number (1-indexed)
            image_index: Sequential index of the image on this page
            ref_type: Type of reference (image, figure, table, etc.)

        Returns:
            URL-safe filename
        """
        # Remove extension from document name
        base_name = Path(document_name).stem

        # Create filename with page and index
        filename = f"{base_name}_p{page_number}_{ref_type}_{image_index}.png"

        # URL-encode special characters for markdown compatibility
        return urllib.parse.quote(filename, safe='')

    def extract_images(
        self,
        ocr_text: str,
        page_image: Image.Image,
        page_number: int,
        document_name: str = "document",
        save_images: bool = True,
    ) -> tuple[str, list[ExtractedImage]]:
        """Extract images from OCR output and replace tags with markdown links.

        Args:
            ocr_text: OCR output text containing reference tags
            page_image: Source page image
            page_number: Page number (1-indexed)
            document_name: Name of the source document (for filenames)
            save_images: Whether to save extracted images to disk

        Returns:
            Tuple of (cleaned_text, list of ExtractedImage objects)
        """
        references = self.find_references(ocr_text)

        if not references:
            # No references found, just clean any remaining ref tags
            cleaned_text = self._clean_remaining_tags(ocr_text)
            return cleaned_text, []

        log.info(
            "extracting_images",
            page=page_number,
            count=len(references),
        )

        extracted_images = []
        replacements = []

        for i, (ref_type, x1, y1, x2, y2, match_start, match_end) in enumerate(references):
            try:
                # Extract the image region
                cropped = self.extract_region(page_image, x1, y1, x2, y2)

                # Generate filename
                filename = self.generate_filename(document_name, page_number, i + 1, ref_type)

                # Create ExtractedImage object
                extracted = ExtractedImage(
                    ref_type=ref_type,
                    page_number=page_number,
                    image_index=i + 1,
                    bbox=(x1, y1, x2, y2),
                    image=cropped,
                )

                # Save image if requested and output directory is set
                if save_images and self.output_dir:
                    file_path = self.output_dir / filename
                    cropped.save(file_path, "PNG")
                    extracted.file_path = file_path
                    log.debug("image_saved", path=str(file_path))

                extracted_images.append(extracted)

                # Create markdown link replacement
                # Use relative path from output_dir if set, otherwise just filename
                if self.output_dir:
                    md_path = f"images/{filename}"
                else:
                    md_path = filename

                markdown_link = f"![{ref_type} {i + 1}]({md_path})"
                replacements.append((match_start, match_end, markdown_link))

            except Exception as e:
                log.warning(
                    "image_extraction_failed",
                    page=page_number,
                    index=i + 1,
                    error=str(e),
                )
                # Replace with placeholder on error
                replacements.append((match_start, match_end, f"[{ref_type} extraction failed]"))

        # Apply replacements in reverse order to preserve positions
        cleaned_text = ocr_text
        for start, end, replacement in reversed(replacements):
            cleaned_text = cleaned_text[:start] + replacement + cleaned_text[end:]

        # Clean any remaining reference tags that weren't matched
        cleaned_text = self._clean_remaining_tags(cleaned_text)

        return cleaned_text, extracted_images

    def _clean_remaining_tags(self, text: str) -> str:
        """Remove any remaining reference/box tags from text.

        Args:
            text: Text possibly containing leftover tags

        Returns:
            Cleaned text
        """
        # Remove incomplete or malformed reference tags
        text = re.sub(r'<[||]ref[||]>[^<]*<[||]/ref[||]>', '', text)
        text = re.sub(r'<[||]box_start[||]>[^<]*<[||]box_end[||]>', '', text)

        # Remove orphaned tags
        text = re.sub(r'<[||]ref[||]>', '', text)
        text = re.sub(r'<[||]/ref[||]>', '', text)
        text = re.sub(r'<[||]box_start[||]>', '', text)
        text = re.sub(r'<[||]box_end[||]>', '', text)

        return text

    def extract_images_batch(
        self,
        pages: list[tuple[str, Image.Image, int]],
        document_name: str = "document",
        save_images: bool = True,
    ) -> Iterator[tuple[str, list[ExtractedImage]]]:
        """Extract images from multiple pages.

        Args:
            pages: List of (ocr_text, page_image, page_number) tuples
            document_name: Name of the source document
            save_images: Whether to save extracted images

        Yields:
            Tuple of (cleaned_text, list of ExtractedImage) for each page
        """
        for ocr_text, page_image, page_number in pages:
            yield self.extract_images(
                ocr_text=ocr_text,
                page_image=page_image,
                page_number=page_number,
                document_name=document_name,
                save_images=save_images,
            )
