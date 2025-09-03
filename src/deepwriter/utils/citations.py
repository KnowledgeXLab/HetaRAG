"""Citation formatting utilities for displaying search results."""

from typing import Dict, Any
from loguru import logger


def print_text_to_image_citation(
    final_images: Dict[int, Dict[str, Any]],
    print_top: bool = True,
) -> None:
    """Prints formatted citations for matched images.

    Args:
        final_images: Dictionary containing matched image information
        print_top: Whether to only print the first citation
    """
    for imageno, image_dict in final_images.items():
        print("\nMatched image path, page number, and page text:")
        print(f"Score: {image_dict['cosine_score']}")
        print(f"File name: {image_dict['file_name']}")
        print(f"Path: {image_dict['image_path']}")
        print(f"Page number: {image_dict['page_num']}")

        if print_top and imageno == 0:
            break


def print_text_to_text_citation(
    final_text: Dict[int, Dict[str, Any]],
    print_top: bool = True,
    chunk_text: bool = True,
) -> None:
    """Prints formatted citations for matched text.

    Args:
        final_text: Dictionary containing matched text information
        print_top: Whether to only print the first citation
        chunk_text: Whether to print individual chunks or full page text
    """
    for textno, text_dict in final_text.items():
        print(f"\nCitation {textno + 1}: Matched text:")
        print(f"Score: {text_dict['cosine_score']}")
        print(f"File name: {text_dict['file_name']}")
        print(f"Page: {text_dict['page_num']}")

        if chunk_text:
            print(f"Chunk number: {text_dict['chunk_id']}")
            print(f"Chunk text: {text_dict['chunk_text']}")
        else:
            print(f"Page text: {text_dict['text']}")

        if print_top and textno == 0:
            break


def post_process_report(
    report_content: str, output_dir: str, image_subdir: str = "images"
) -> str:
    """Post-process the report to ensure it is formatted correctly.

    Args:
        report_content: The report content to post-process
        output_dir: The directory path to save the post-processed report
        image_subdir: The subdirectory name to save the images

    Returns:
        The post-processed report content with updated image paths
    """
    import re
    import shutil
    from pathlib import Path

    # Extract image references and copy images to output directory
    markdown_image_pattern = r"!\[.*\](.*)"
    source_image_paths = re.findall(markdown_image_pattern, report_content)

    # Create output directories
    report_output_dir = Path(output_dir)
    report_output_dir.mkdir(parents=True, exist_ok=True)

    images_output_dir = report_output_dir / image_subdir
    images_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image reference
    processed_content = report_content
    for source_path_str in source_image_paths:
        # Remove enclosing parentheses from markdown image syntax
        source_path = source_path_str[1:-1]

        # Construct destination path
        dest_path = images_output_dir / Path(source_path).name

        # Copy image file
        shutil.copy(source_path, dest_path)
        logger.info(f"Copied image from {source_path} to {dest_path}")

        # Update image reference in report
        relative_path = Path(image_subdir) / Path(source_path).name
        processed_content = processed_content.replace(source_path, str(relative_path))

    return processed_content
