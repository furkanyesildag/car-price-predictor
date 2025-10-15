import argparse
import glob
import io
import json
import os
import sys
from typing import Generator, Iterable, List, Optional


def iter_cars_from_file(path: str) -> Generator[dict, None, None]:
	"""Yield car objects from a large JSON text file without loading entire file.

	The expected file structure is an object that contains a key "cars" whose value
	is a JSON array of car objects. The file may be a single very long line.

	This function scans for the substring '"cars":[' and then streams individual
	JSON objects inside that array using a brace/quote aware state machine, and
	decodes each object with json.loads.
	"""

	with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
		data = f.read()

		start_key = '"cars":['
		start_idx = data.find(start_key)
		if start_idx == -1:
			return

		idx = start_idx + len(start_key)
		length = len(data)
		in_string = False
		escaped = False
		brace_depth = 0
		current = []

		while idx < length:
			ch = data[idx]
			if in_string:
				if escaped:
					escaped = False
				else:
					if ch == "\\":
						escaped = True
					elif ch == '"':
						in_string = False
				current.append(ch)
				idx += 1
				continue

			if ch == '"':
				in_string = True
				current.append(ch)
				idx += 1
				continue

			if ch == "{" and brace_depth == 0:
				# start of an object
				brace_depth = 1
				current = [ch]
				idx += 1
				continue

			if brace_depth > 0:
				current.append(ch)
				if ch == "{":
					brace_depth += 1
				elif ch == "}":
					brace_depth -= 1
					if brace_depth == 0:
						try:
							obj_text = "".join(current)
							yield json.loads(obj_text)
						except json.JSONDecodeError:
							pass
						current = []
				idx += 1
				continue

			# end of cars array
			if ch == "]":
				break

			idx += 1


def normalize_car(car: dict) -> dict:
	return {
		"_id": car.get("_id"),
		"adId": car.get("adId"),
		"brand": car.get("brand"),
		"series": car.get("series"),
		"model": car.get("model"),
		"year": car.get("year"),
		"km": car.get("km"),
		"price": car.get("price"),
		"createdAt": car.get("createdAt"),
		"updatedAt": car.get("updatedAt"),
		"__v": car.get("__v"),
	}


def write_csv(rows: Iterable[dict], out_path: str) -> int:
	import csv
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fieldnames = [
		"_id",
		"adId",
		"brand",
		"series",
		"model",
		"year",
		"km",
		"price",
		"createdAt",
		"updatedAt",
		"__v",
	]
	count = 0
	with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)
			count += 1
	return count


def gather_rows(paths: List[str]) -> Generator[dict, None, None]:
	for path in paths:
		for car in iter_cars_from_file(path):
			yield normalize_car(car)


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Extract cars from JSON text files to CSV")
	parser.add_argument(
		"--inputs",
		nargs="*",
		help="Input file paths (defaults to *.txt and *.tx)",
	)
	parser.add_argument(
		"--output",
		default=os.path.join("data", "cars.csv"),
		help="Output CSV path",
	)
	args = parser.parse_args(argv)

	if args.inputs:
		input_paths = args.inputs
	else:
		cwd = os.getcwd()
		input_paths = sorted(set(glob.glob(os.path.join(cwd, "*.txt")) + glob.glob(os.path.join(cwd, "*.tx"))))

	if not input_paths:
		print("No input files found.", file=sys.stderr)
		return 1

	row_iter = gather_rows(input_paths)
	count = write_csv(row_iter, args.output)
	print(f"Wrote {count} rows to {args.output}")
	return 0


if __name__ == "__main__":
	sys.exit(main())


