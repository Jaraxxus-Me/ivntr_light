"""Download required data for blocked_stacking scenarios.

This script downloads a single data.zip file from Google Drive containing:
- 1024_sc1_pred_nets: Scenario 1 predicate networks
- 1024_sc1_skills: Scenario 1 skill checkpoints
- 1024_sc2_pred_nets: Scenario 2 predicate networks
- 1024_sc2_skills: Scenario 2 skill checkpoints
- logs: Training logs
- training_data: Planner datasets

Usage:
    python scripts/download_sc1_2.py
    python scripts/download_sc1_2.py --file-id <google_drive_id_or_url>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

try:
    import gdown
except ImportError:
    print("Error: gdown library not found. Install with: pip install gdown")
    sys.exit(1)


def extract_file_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from URL or return ID if already extracted.

    Args:
        url_or_id: Either a Google Drive URL or a file ID

    Returns:
        The file ID string

    Examples:
        >>> extract_file_id("https://drive.google.com/file/d/xx/view?usp=sharing")
        '1lWfzy6f'
        >>> extract_file_id("1lWfzy6f")
        '1lWfzy6f'
    """
    # If it's already just an ID (no slashes or domains), return it
    if "/" not in url_or_id and "drive.google.com" not in url_or_id:
        return url_or_id

    # Extract from /d/FILE_ID/ or /d/FILE_ID? patterns
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)

    # Extract from id=FILE_ID pattern
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract file ID from: {url_or_id}")


def download_file_from_google_drive(
    file_id: str, destination: Path, verbose: bool = True
) -> None:
    """Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID
        destination: Path where the file should be saved
        verbose: Whether to print progress messages
    """
    if verbose:
        print(f"Downloading from Google Drive (ID: {file_id})")
        print(f"  -> {destination}")

    # Create parent directory if it doesn't exist
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Build Google Drive URL
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        # Use gdown to download the file (handles large files automatically)
        gdown.download(url, str(destination), quiet=not verbose, fuzzy=True)

        if verbose and destination.exists():
            size_mb = destination.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {size_mb:.2f} MB")

    except Exception as e:
        raise RuntimeError(f"Failed to download file {file_id}: {e}") from e


def extract_zip(
    zip_path: Path,
    extract_to: Path,
    target_dirs: list[str],
    verbose: bool = True,
) -> None:
    """Extract a zip file and ensure target directories are at the root level.

    If the zip contains a single parent folder with the target directories inside,
    this function will move the contents up to the extract_to level.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to
        target_dirs: List of expected directory names
        verbose: Whether to print progress messages
    """
    if verbose:
        print(f"Extracting {zip_path.name}")
        print(f"  -> {extract_to}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Extract to a temporary location first
        temp_extract = extract_to / ".temp_extract"
        temp_extract.mkdir(exist_ok=True)

        zip_ref.extractall(temp_extract)
        file_count = len(zip_ref.namelist())

        if verbose:
            print(f"  Extracted {file_count} files")

    # Check if we have a nested structure (single parent folder)
    extracted_items = list(temp_extract.iterdir())

    # If there's only one directory and it contains our target directories
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        parent_folder = extracted_items[0]
        # Check if target directories are inside this parent folder
        has_targets_inside = any(
            (parent_folder / target_dir).exists() for target_dir in target_dirs
        )

        if has_targets_inside:
            if verbose:
                print(
                    f"  Found nested structure in '{parent_folder.name}', flattening..."
                )

            # Move contents from parent folder to extract_to
            for item in parent_folder.iterdir():
                dest = extract_to / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(extract_to))

            # Remove the now-empty parent folder and temp directory
            shutil.rmtree(temp_extract)
            if verbose:
                print(f"  Moved contents to root level")
        else:
            # No nesting, just move everything to extract_to
            for item in temp_extract.iterdir():
                dest = extract_to / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(extract_to))
            shutil.rmtree(temp_extract)
    else:
        # Multiple items at root or files - move everything
        for item in temp_extract.iterdir():
            dest = extract_to / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(extract_to))

        # Clean up temp directory
        shutil.rmtree(temp_extract)


def download_sc1_data(
    file_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
    verbose: bool = True,
) -> None:
    """Download all required data for blocked_stacking scenarios.

    This function downloads a single data.zip file from Google Drive containing:
    - 1024_sc1_pred_nets, 1024_sc1_skills, 1024_sc2_pred_nets, 1024_sc2_skills
    - logs and training_data directories

    Args:
        file_id: Google Drive file ID or URL. If None, uses default.
        repo_root: Root directory of the repository.
        verbose: Whether to print progress messages

    Raises:
        FileNotFoundError: If repo_root cannot be determined
        Exception: If download or extraction fails
    """
    # Determine repository root
    if repo_root is None:
        # Assume script is in scripts/ directory
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent

    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")

    if verbose:
        print(f"Repository root: {repo_root}")
        print()

    # Default Google Drive file ID
    if file_id is None:
        file_id = "https://drive.google.com/file/d/12PWBLOtiNxy7iFiURX1vkJq4Pc26UMng/view?usp=sharing"

    # Extract file ID from URL if necessary
    file_id = extract_file_id(file_id)

    # Define paths
    zip_path = repo_root / "data.zip"

    try:
        if verbose:
            print(f"{'=' * 60}")
            print("Downloading data.zip")
            print(f"{'=' * 60}")

        # Check if any target directories already exist
        target_dirs = [
            "top_down_pred_nets",
            "training_data",
        ]
        existing_dirs = [d for d in target_dirs if (repo_root / d).exists()]

        if existing_dirs:
            if verbose:
                print(f"  Found existing directories: {', '.join(existing_dirs)}")
                print("  Skipping download...")
                print()
            return

        # Download from Google Drive
        download_file_from_google_drive(file_id, zip_path, verbose=verbose)

        # Extract with automatic flattening if nested
        extract_zip(zip_path, repo_root, target_dirs, verbose=verbose)

        # Clean up zip file
        zip_path.unlink()
        if verbose:
            print(f"  Removed {zip_path.name}")
            print()

    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

    if verbose:
        print(f"{'=' * 60}")
        print("Download complete!")
        print(f"{'=' * 60}")
        print()
        print("Extracted directories:")
        for dir_name in target_dirs:
            if (repo_root / dir_name).exists():
                print(f"  - {dir_name}")
        print()
        print("You can now run tests with:")
        print(
            "  pytest tests/approaches/test_discovered_pred_skills.py::"
            "test_loading_learned_predicates_blocked_stacking -xvs"
        )


def main() -> int:
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download required data for blocked_stacking scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with default settings (uses hardcoded Google Drive ID)
  python scripts/download_sc1_2.py

  # Use custom Google Drive file ID
  python scripts/download_sc1_2.py --file-id 1lWfzy6f_uVU1SRCbi1Xyl0euwWrIq5tn

  # Can also pass full URL
  python scripts/download_sc1_2.py --file-id \\
    https://drive.google.com/file/d/1zG7.../view?usp=sharing

  # Quiet mode
  python scripts/download_sc1_2.py --quiet

The data.zip file contains:
  - top_down_pred_nets: predicate networks
  - training_data: Planner datasets
        """,
    )

    parser.add_argument(
        "--file-id",
        type=str,
        default="https://drive.google.com/file/d/1DpQhY1M7jtRiA79-rsK68y7Q6W3Xnl0t/view?usp=sharing",
        help="Google Drive file ID or URL for data.zip",
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory " "(default: auto-detect from script location)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    try:
        download_sc1_data(
            file_id=args.file_id,
            repo_root=args.repo_root,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())