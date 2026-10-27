
import csv
import json

with open("reviews.csv", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

with open("reviews.json", encoding="utf-8") as file:
    reviews = json.load(file)

#Here, the small lambda function tells sorted to compare each row by
#its score. reverse=True puts the highest score first.

best_first = sorted(rows, key=lambda row: float(row["score"]), reverse=True)