import scandir
from typing import Iterator, Tuple

def walk_dir(dir: str) -> Iterator[Tuple[str, str]]:
	"""
	Walk through a directory and its immediate subdirectories,
	yielding file paths and their relative paths.

	Args:
		dir (str): Path to the directory to scan.

	Yields:
		Iterator[Tuple[str, str]]: Tuples containing the full path
		to the file and its relative path.
	"""
	for entry in scandir.scandir(dir):
		if entry.is_dir():
			for file in scandir.scandir(entry.path):
				if file.is_file():
					yield file.path, f"{entry.name}/{file.name}"
		elif entry.is_file():
			yield entry.path, entry.name

