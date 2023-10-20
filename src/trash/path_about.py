from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
print(type(PROJECT_DIR.as_posix()))