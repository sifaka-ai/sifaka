from pathlib import Path
import sys

# To obtain modules from sifaka
if (Path(__file__).resolve().parents[1]).exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[1]))
