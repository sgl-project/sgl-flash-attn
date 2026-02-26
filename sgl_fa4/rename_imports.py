import os
import re
import argparse
from pathlib import Path

def rename_imports(target_dir, old_pkg, new_pkg, dry_run=False):
    # Regex to match 'import abc.cde' or 'from abc.cde'
    # It ensures the match is at the start of a line or after whitespace
    pattern = re.compile(rf'(\bimport\s+|\bfrom\s+){re.escape(old_pkg)}(\b|\.)')
    replacement = rf'\1{new_pkg}\2'

    print(f"Scanning {target_dir} for imports of '{old_pkg}'...")

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Perform the replacement
                new_content, count = pattern.subn(replacement, content)

                if count > 0:
                    if not dry_run:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"  Fixed {count} imports in: {file_path}")
                    else:
                        print(f"  Would fix {count} imports in: {file_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Rename Python package imports recursively.")
    
    # Define the arguments
    parser.add_argument("-d", "--target-dir", required=True, help="Target directory (e.g. ./src)")
    parser.add_argument("-o", "--old-pkg", required=True, help="Old package string (e.g. abc.cde)")
    parser.add_argument("-n", "--new-pkg", required=True, help="New package string (e.g. xyz.lmn)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without saving them")
    args = parser.parse_args()

    # Verify directory exists
    if not os.path.isdir(args.target_dir):
        print(f"❌ Error: Directory '{args.target_dir}' does not exist.")
        exit(1)

    rename_imports(args.target_dir, args.old_pkg, args.new_pkg, args.dry_run)
