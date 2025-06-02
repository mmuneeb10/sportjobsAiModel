import fitz  # PyMuPDF
import json
import numpy as np
from collections import defaultdict


# import pdfplumber

# def extract_pdf_content(pdf_path):
#     """
#     Extract and print both formatted content and raw text from a PDF file
#     using pdfplumber.

#     Args:
#         pdf_path (str): Path to the PDF file
#     """
#     try:
#         # Open the PDF file
#         with pdfplumber.open(pdf_path) as pdf:
#             print("\n" + "="*50)
#             print(f"PDF INFORMATION")
#             print(f"Total Pages: {len(pdf.pages)}")
#             print("="*50 + "\n")

#             # Process each page
#             for i, page in enumerate(pdf.pages):
#                 print(f"\n{'='*30} PAGE {i+1} {'='*30}\n")

#                 # Extract text while preserving format
#                 print("FORMATTED TEXT:")
#                 print("-"*50)
#                 text = page.extract_text()
#                 if text:
#                     print(text)
#                 else:
#                     print("No text content found on this page.")
#                 print("-"*50)

#                 # Extract raw text
#                 print("\nRAW TEXT:")
#                 print("-"*50)
#                 chars = page.chars
#                 if chars:
#                     raw_text = "".join([char["text"] for char in chars])
#                     print(raw_text)
#                 else:
#                     print("No raw text content found on this page.")
#                 print("-"*50)

#                 # Print page dimensions
#                 print(f"\nPage Dimensions: Width={page.width}, Height={page.height}")

#                 print(f"\n{'='*70}\n")

#     except Exception as e:
#         print(f"Error processing PDF: {str(e)}")


# def extract_pdf_content(pdf_path):

#     structured_data = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page_number, page in enumerate(pdf.pages):
#             # Extract words with their bounding boxes
#             words = page.extract_words(
#                 use_text_flow=True, keep_blank_chars=True  # respects reading order
#             )

#             for word in words:
#                 structured_data.append(
#                     {
#                         "page": page_number + 1,
#                         "text": word.get("text"),
#                         "x0": word.get("x0"),  # left
#                         "x1": word.get("x1"),  # right
#                         "top": word.get("top"),  # y-position
#                         "bottom": word.get("bottom"),
#                         "width": word.get("x1") - word.get("x0"),
#                         "height": word.get("bottom") - word.get("top"),
#                         "font_size": round(
#                             word.get("bottom") - word.get("top"), 2
#                         ),  # rough font size
#                     }
#                 )
#     print("PDF STRUCTURED FORMAT IS ", structured_data)


def extract_pdf_content(pdf_path):
    """
    Step 1: Document Processing - Extract CV formatting information from PDF

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        dict: Structured representation of CV with formatting information
    """
    # Initialize document structure
    cv_data = {
        "metadata": {},
        "pages": [],
        "fonts": {},
        "sections": [],
        "lists": [],
        "visual_elements": [],
        "format_features": {},
    }

    try:
        # Open PDF document
        doc = fitz.open(pdf_path)

        # Extract basic metadata
        cv_data["metadata"] = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": len(doc),
            "file_size_kb": round(doc.tobytes().__len__() / 1024, 2),
        }

        # Track document-wide formatting elements
        all_fonts = {}
        section_candidates = []
        line_heights = []
        line_spacings = []
        left_margins = []
        bullet_points = []
        prev_line_bottom = None

        # Process each page
        for page_num, page in enumerate(doc):
            page_data = {
                "page_number": page_num + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "content": [],
            }

            # Get text with detailed formatting
            text_dict = page.get_text("dict")

            # Process text blocks
            for block in text_dict.get("blocks", []):
                # The Block type 1  is equal to text and we have other types as images etc
                if block["type"] == 0:  # Text block
                    for line in block.get("lines", []): # It is to get each line text 
                        line_bbox = line.get("bbox", [0, 0, 0, 0])
                        line_top = line_bbox[1]
                        line_bottom = line_bbox[3]
                        line_height = line_bottom - line_top
                        line_left = line_bbox[0]

                        # Track line spacing
                        if prev_line_bottom is not None:
                            line_spacing = line_top - prev_line_bottom
                            if 0 < line_spacing < 50:  # Filter out unreasonable values (It is to check the line spacing between two lines)
                                line_spacings.append(line_spacing) 

                        prev_line_bottom = line_bottom
                        line_heights.append(line_height) # For consistency saving the line heights and line margins
                        left_margins.append(line_left)

                        # Extract text and formatting info
                        line_text = ""
                        line_data = {
                            "bbox": line_bbox,
                            "left": line_left,
                            "top": line_top,
                            "text": "",
                            "spans": [],
                        }

                        # Process spans within line
                        for span in line.get("spans", []):
                            span_text = span.get("text", "").strip()
                            font_name = span.get("font", "unknown")
                            font_size = span.get("size", 0)
                            font_flags = span.get("flags", 0)
                            color = span.get("color", 0)

                            # Determine text style
                            is_bold = bool(font_flags & 16)
                            is_italic = bool(font_flags & 2)

                            # Add to line text
                            line_text += span_text

                            # Add span data
                            line_data["spans"].append(
                                {
                                    "text": span_text,
                                    "font_name": font_name,
                                    "font_size": font_size,
                                    "is_bold": is_bold,
                                    "is_italic": is_italic,
                                    "color": color,
                                    "bbox": span.get("bbox", [0, 0, 0, 0]),
                                }
                            )

                            # Track font usage
                            # Adding fonts like all the fonts which are used in the cv
                            font_key = f"{font_name}_{font_size}_{is_bold}_{is_italic}"
                            if font_key not in all_fonts:
                                all_fonts[font_key] = {
                                    "name": font_name,
                                    "size": font_size,
                                    "is_bold": is_bold,
                                    "is_italic": is_italic,
                                    "count": 0,
                                    "samples": [],
                                }

                            all_fonts[font_key]["count"] += 1
                            # Adding the fonts sample for further usage
                            if (
                                len(all_fonts[font_key]["samples"]) < 3
                            ):  # Keep a few samples
                                all_fonts[font_key]["samples"].append(span_text)

                            # Identify potential section headers
                            if (is_bold or font_size > 11) and len(span_text) < 100:
                                section_candidates.append(
                                    {
                                        "text": span_text,
                                        "page": page_num + 1,
                                        "position": (line_top, line_left),
                                        "font_size": font_size,
                                        "is_bold": is_bold,
                                    }
                                )

                        # Check for bullet points
                        if line_text.strip().startswith(
                            ("•", "-", "✓", "✔", "➢", "*")
                        ) or (
                            len(line_text) > 2
                            and line_text[0].isdigit()
                            and line_text[1] == "."
                        ):
                            bullet_points.append(
                                {
                                    "text": line_text,
                                    "page": page_num + 1,
                                    "position": (line_top, line_left),
                                    "indent": line_left,
                                }
                            )

                        # Add line data
                        line_data["text"] = line_text
                        page_data["content"].append(line_data)

                # Handle images and other visual elements
                elif block["type"] == 1:  # Image
                    cv_data["visual_elements"].append(
                        {
                            "type": "image",
                            "page": page_num + 1,
                            "bbox": block.get("bbox", [0, 0, 0, 0]),
                        }
                    )

            # Add page data
            cv_data["pages"].append(page_data)

        # Process and analyze collected data
        cv_data["fonts"] = sorted(
            all_fonts.values(), key=lambda x: (x["size"], x["count"]), reverse=True
        )

        # Identify sections based on formatting
        cv_data["sections"] = identify_sections(section_candidates)

        # Group bullet points into lists
        cv_data["lists"] = identify_lists(bullet_points)

        # Calculate format features
        cv_data["format_features"] = calculate_format_features(
            cv_data["fonts"],
            line_heights,
            line_spacings,
            left_margins,
            cv_data["pages"],
        )

        return cv_data

    except Exception as e:
        return {"error": str(e)}


def identify_sections(candidates):
    """Identify document sections based on formatting cues"""
    if not candidates:
        return []

    # Sort by page and position
    sorted_candidates = sorted(candidates, key=lambda x: (x["page"], x["position"][0]))

    # Group similar headers (might be subsections)
    sections = []

    # Calculate average font size of potential headers
    font_sizes = [c["font_size"] for c in sorted_candidates]
    avg_font_size = sum(font_sizes) / len(font_sizes)

    # Identify main sections vs subsections
    for candidate in sorted_candidates:
        section_type = "main" if candidate["font_size"] >= avg_font_size else "sub"
        if candidate["is_bold"]:
            section_type = "main" if section_type == "main" else "sub"

        sections.append(
            {
                "text": candidate["text"].strip(),
                "page": candidate["page"],
                "position": candidate["position"],
                "font_size": candidate["font_size"],
                "is_bold": candidate["is_bold"],
                "type": section_type,
            }
        )

    return sections


def identify_lists(bullet_points):
    """Group bullet points into lists based on positioning and indentation"""
    if not bullet_points:
        return []

    lists = []
    current_list = None

    # Sort by page and position
    sorted_bullets = sorted(bullet_points, key=lambda x: (x["page"], x["position"][0]))

    for bullet in sorted_bullets:
        if current_list is None:
            # Start a new list
            current_list = {
                "page": bullet["page"],
                "indent": bullet["indent"],
                "items": [bullet],
            }
        elif (
            current_list["page"] == bullet["page"]
            and abs(current_list["indent"] - bullet["indent"]) < 5
        ):
            # Continuing the same list
            current_list["items"].append(bullet)
        else:
            # End current list and start a new one
            lists.append(current_list)
            current_list = {
                "page": bullet["page"],
                "indent": bullet["indent"],
                "items": [bullet],
            }

    # Add the last list if any
    if current_list is not None:
        lists.append(current_list)

    return lists


def calculate_format_features(fonts, line_heights, line_spacings, left_margins, pages):
    """Calculate features needed for format evaluation"""
    features = {}

    # Font consistency
    font_counts = [font["count"] for font in fonts]
    total_fonts = sum(font_counts)
    primary_font_ratio = max(font_counts) / total_fonts if total_fonts > 0 else 0

    features["font_metrics"] = {
        "unique_font_count": len(fonts),
        "primary_font_ratio": primary_font_ratio,
        "font_variety_score": (
            min(1.0, 3 / len(fonts)) if fonts else 0
        ),  # 3 fonts ideal (1.0)
    }

    # Spacing consistency
    if line_heights:
        features["spacing_metrics"] = {
            "line_height_avg": np.mean(line_heights),
            "line_height_std": np.std(line_heights),
            "line_height_consistency": 1.0 / (1.0 + np.std(line_heights)),
            "line_spacing_avg": np.mean(line_spacings) if line_spacings else 0,
            "line_spacing_std": np.std(line_spacings) if line_spacings else 0,
            "line_spacing_consistency": (
                1.0 / (1.0 + np.std(line_spacings)) if line_spacings else 0
            ),
        }
    else:
        features["spacing_metrics"] = {
            "line_height_consistency": 0,
            "line_spacing_consistency": 0,
        }

    # Alignment consistency
    if left_margins:
        # Group margins into clusters to see how many alignment points are used
        margin_clusters = {}
        threshold = 5  # pixels

        for margin in left_margins:
            rounded = round(margin / threshold) * threshold
            margin_clusters[rounded] = margin_clusters.get(rounded, 0) + 1

        strongest_alignment = max(margin_clusters.values()) / len(left_margins)

        features["alignment_metrics"] = {
            "left_margin_std": np.std(left_margins),
            "alignment_consistency": strongest_alignment,
            "alignment_points": len(margin_clusters),
        }
    else:
        features["alignment_metrics"] = {"alignment_consistency": 0}

    # Density metrics
    text_area = 0
    total_area = 0

    for page in pages:
        page_area = page["width"] * page["height"]
        total_area += page_area

        # Rough estimation of text area
        text_content_area = 0
        for content in page["content"]:
            bbox = content["bbox"]
            if bbox:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                text_content_area += width * height

        text_area += text_content_area

    features["density_metrics"] = {
        "text_density": text_area / total_area if total_area > 0 else 0,
        "whitespace_ratio": 1 - (text_area / total_area) if total_area > 0 else 0,
    }

    return features


# Example usage
if __name__ == "__main__":
    pdf_path = "/home/faizan/workspace/Sports Job Model Training/MuhammadMuneebResume.pdf"
    # pdf_path = "/home/faizan/workspace/Sports Job Model Training/BadCV.pdf"
    cv_format_data = extract_pdf_content(pdf_path)

    # Save to JSON for inspection
    with open("cv_format_data.json", "w", encoding="utf-8") as f:
        json.dump(cv_format_data, f, indent=2)

    print("CV format information extracted successfully.")
    print(f"Found {len(cv_format_data['fonts'])} unique fonts")
    print(f"Identified {len(cv_format_data['sections'])} potential sections")
    print(f"Detected {len(cv_format_data['lists'])} list structures")

    # Print key format metrics
    metrics = cv_format_data["format_features"]
    print("\nFormat Metrics:")
    print(f"Font Variety Score: {metrics['font_metrics']['font_variety_score']:.2f}")
    print(f"Primary Font Usage: {metrics['font_metrics']['primary_font_ratio']:.1%}")
    print(
        f"Alignment Consistency: {metrics['alignment_metrics']['alignment_consistency']:.1%}"
    )
    print(f"Text Density: {metrics['density_metrics']['text_density']:.1%}")
