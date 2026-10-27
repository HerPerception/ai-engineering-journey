from dataclasses import dataclass
@dataclass(frozen=True)
class Record:
    features: tuple[float, ...]
    label: int

def row_to_record(row, feature_names):
    values = tuple(float(row[name]) for name in feature_names)
    return Record(values, int(row["label"]))