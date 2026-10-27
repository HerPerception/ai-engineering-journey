# A type hint describes what kind of value a function expects and 
# returns. It helps people and editor tools understand the code, 
# but Python generally doesn’t enforce the hint when the program 
# runs.
def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def main() -> None:
    print(mean([1.0, 2.0, 4.0]))

if __name__ == "__main__":
    main()
