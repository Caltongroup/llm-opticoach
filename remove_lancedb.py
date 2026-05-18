#!/usr/bin/env python3
import shutil
import re
from datetime import datetime
from pathlib import Path

def create_backup(filepath):
    if filepath.exists():
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{filepath.name}.{timestamp}.bak"
        shutil.copy2(filepath, backup_path)
        print(f"Backup created: {backup_path}")

def main():
    main_file = Path("main.py")
    create_backup(main_file)
    
    content = main_file.read_text()

    # Remove import
    content = re.sub(r'^\s*import lancedb.*\n', '', content, flags=re.MULTILINE)
    
    # Remove LANCEDB_PATH line
    content = re.sub(r'^\s*LANCEDB_PATH\s*=.*\n', '', content, flags=re.MULTILINE)

    # Remove the function
    content = re.sub(r'def _store_tuning_insight_to_lancedb\(.*?(?=\n\w|\Z)', '', content, flags=re.DOTALL)

    # Remove calls to the function
    content = re.sub(r'_store_tuning_insight_to_lancedb\(.*?\)', '# LanceDB call removed', content)

    main_file.write_text(content)
    print("LanceDB removed from main.py")

if __name__ == "__main__":
    main()
