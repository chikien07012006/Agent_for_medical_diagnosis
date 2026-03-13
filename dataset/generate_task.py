"""
convert_to_confidx_tasks.py
Reads a JSON file containing clinical cases (in the provided format) and
outputs four JSON files in the Alpaca instruction format required by ConfiDx.
"""

import json
import os
from typing import List, Dict, Any

# ========== CONFIGURATION ==========
INPUT_JSON = "path"               # Path to your input JSON file
OUTPUT_DIR = "path"     # Where to save the task files
# ====================================

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Instruction templates
INSTRUCTION_DIAGNOSIS = "You are an experienced doctor. Given a patient's clinical note, identify the most likely disease."
INSTRUCTION_EXPLANATION = "You are an experienced doctor. Given a patient's clinical note, provide a step-by-step diagnostic explanation."
INSTRUCTION_UNCERTAINTY_LABEL = "You are an experienced doctor. Given a patient's clinical note, determine the level of diagnostic uncertainty: Certain, Uncertain, or Unknown."
INSTRUCTION_UNCERTAINTY_EXPLANATION = "You are an experienced doctor. Given a patient's clinical note, explain why the diagnosis is uncertain."

# Mapping from your uncertainty level to the paper's labels
UNCERTAINTY_MAP = {
    "low_uncertainty": "Certain",
    "medium_uncertainty": "Uncertain",
    "high_uncertainty": "Uncertain",
    "unknown": "Unknown"
}

def load_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Load cases from a JSON file that is either a JSON array or JSON Lines.
    """
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to load as a single JSON array first
        try:
            data = json.load(f)
            if isinstance(data, list):
                cases = data
            else:
                # If it's a single object, wrap it in a list
                cases = [data]
        except json.JSONDecodeError:
            # Not a valid JSON array, try reading line by line (JSON Lines)
            f.seek(0)  # go back to start
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cases.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    return cases

def generate_diagnostic_explanation(case: Dict[str, Any]) -> str:
    """
    Generate a step-by-step diagnostic explanation from the case data.
    This is a simple template; you can replace it with a more sophisticated one.
    """
    primary = case['labels']['primary_diagnosis']
    evidence = case['labels']['evidence']
    differentials = case['labels']['differential_diagnoses']
    
    lines = [f"The patient's most likely diagnosis is {primary}."]
    
    if evidence:
        lines.append("Key findings supporting this diagnosis include:")
        for item in evidence:
            lines.append(f"- {item}")
    else:
        lines.append("No specific abnormal findings were recorded.")
    
    if differentials:
        lines.append("Differential diagnoses considered: " + ", ".join(differentials[:3]))
        if len(differentials) > 3:
            lines.append("... and others.")
    
    return "\n".join(lines)

def main():
    # Load cases
    cases = load_cases(INPUT_JSON)
    print(f"Loaded {len(cases)} cases from {INPUT_JSON}")

    if not cases:
        print("No cases loaded. Exiting.")
        return

    # Prepare lists for each task
    task1 = []   # diagnosis
    task2 = []   # diagnostic explanation
    task3 = []   # uncertainty label
    task4 = []   # uncertainty explanation

    for case in cases:
        note = case['input_text']
        labels = case['labels']
        
        # Task 1: Diagnosis
        task1.append({
            "instruction": INSTRUCTION_DIAGNOSIS,
            "input": note,
            "output": labels['primary_diagnosis']
        })
        
        # Task 2: Diagnostic explanation
        explanation = generate_diagnostic_explanation(case)
        task2.append({
            "instruction": INSTRUCTION_EXPLANATION,
            "input": note,
            "output": explanation
        })
        
        # Task 3: Uncertainty label
        uncertainty_level = labels['uncertainty_level']
        paper_label = UNCERTAINTY_MAP.get(uncertainty_level, "Unknown")
        task3.append({
            "instruction": INSTRUCTION_UNCERTAINTY_LABEL,
            "input": note,
            "output": paper_label
        })
        
        # Task 4: Uncertainty explanation (only if reasons exist, else empty)
        reasons = labels.get('uncertainty_reasons', [])
        output_uncert = " ".join(reasons) if reasons else ""
        task4.append({
            "instruction": INSTRUCTION_UNCERTAINTY_EXPLANATION,
            "input": note,
            "output": output_uncert
        })

    # Save each task file
    with open(os.path.join(OUTPUT_DIR, "task1_diagnosis_test.json"), "w", encoding='utf-8') as f:
        json.dump(task1, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(OUTPUT_DIR, "task2_explanation_test.json"), "w", encoding='utf-8') as f:
        json.dump(task2, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(OUTPUT_DIR, "task3_uncertainty_label_test.json"), "w", encoding='utf-8') as f:
        json.dump(task3, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(OUTPUT_DIR, "task4_uncertainty_explanation_test.json"), "w", encoding='utf-8') as f:
        json.dump(task4, f, indent=2, ensure_ascii=False)

    print(f"Task files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()