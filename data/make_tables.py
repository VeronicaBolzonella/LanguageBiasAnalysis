import json
import pandas as pd

def make_table(file, output_file):
    # Load JSON
    with open(file, "r") as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Optional: rename columns for publication style
    df = df.rename(columns={
        "Gender_orig": "Word",
        "Gender_new": "Original Gender Score",
        "new": "Finetuned Gender Score",
        "Age_orig": "Original Age Score",
        "Age_new": "Finetuned Age Score",
        "Class_orig": "Original Class Score",
        "Class_new": "Finetuned Class Score"
    })

    latex_str = df.to_latex(index=False, float_format="%.4f")

    with open(output_file, "w") as f:
        f.write(latex_str)

    print("Saved")


if __name__ == "__main__":    
    make_table("data/results/adjectives_all_biases.json", "data/tables/table_adjectives.tex")
    make_table("data/results/animals_all_biases.json", "data/tables/table_animals.tex")
    make_table("data/results/professions_all_biases.json", "data/tables/table_professions.tex")