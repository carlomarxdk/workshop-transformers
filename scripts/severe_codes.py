"""The severe-disease outcome definition for workshop A02.

An explicit code list, not a regex, so it can be put on a slide and argued with.
The exclusions matter as much as the inclusions -- see planning/a02-task-design.md.
"""

SEVERE = {
    "Cardiovascular": [
        "Myocardial infarction (disorder)",
        "Acute ST segment elevation myocardial infarction (disorder)",
        "Acute non-ST segment elevation myocardial infarction (disorder)",
        "Preinfarction syndrome (disorder)",
        "Cerebrovascular accident (disorder)",
        "Chronic congestive heart failure (disorder)",
        "Heart failure (disorder)",
        "Acute pulmonary embolism (disorder)",
        "Acute deep venous thrombosis (disorder)",
    ],
    "Cancer": [
        "Malignant neoplasm of breast (disorder)",
        "Malignant neoplasm of colon (disorder)",
        "Overlapping malignant neoplasm of colon (disorder)",
        "Primary malignant neoplasm of colon (disorder)",
        "Metastatic malignant neoplasm to colon (disorder)",
        "Non-small cell lung cancer (disorder)",
        "Non-small cell carcinoma of lung  TNM stage 1 (disorder)",
        "Small cell carcinoma of lung (disorder)",
        "Primary small cell malignant neoplasm of lung  TNM stage 1 (disorder)",
        "Neoplasm of prostate (disorder)",
        "Carcinoma in situ of prostate (disorder)",
        "Metastatic malignant neoplasm to prostate (disorder)",
        "Acute myeloid leukemia (disorder)",
    ],
    "Renal failure": [
        "End-stage renal disease (disorder)",
        "Chronic kidney disease stage 4 (disorder)",
        "Kidney transplant failure and rejection (disorder)",
        "Acute renal failure on dialysis (disorder)",
        "Postoperative renal failure (disorder)",
    ],
    "Respiratory failure": [
        "Acute respiratory failure (disorder)",
        "Acute respiratory distress syndrome (disorder)",
        "Pulmonary emphysema (disorder)",
        "Chronic obstructive bronchitis (disorder)",
    ],
    "Sepsis / shock": [
        "Sepsis (disorder)",
        "Septic shock (disorder)",
        "Shock (disorder)",
        "Sepsis caused by virus (disorder)",
        "Sepsis caused by Staphylococcus aureus (disorder)",
        "Sepsis caused by Pseudomonas (disorder)",
    ],
    "Neurodegenerative": [
        "Familial Alzheimer's disease of early onset (disorder)",
    ],
}

# Codes a reasonable person might include, and the reason not to.
# The rule: a code that records the health system LOOKING for a disease is not the disease.
EXCLUDED = {
    "Suspected lung cancer (situation)":
        "work-up artefact -- records that someone looked, not what was found",
    "Suspected prostate cancer (situation)":
        "work-up artefact",
    "Awaiting transplantation of kidney (situation)":
        "administrative status, downstream of end-stage renal disease",
    "Died in hospice (finding)":
        "a death marker; including it would leak the outcome",
    "History of myocardial infarction (situation)":
        "history code -- refers to the past, not a new event",
    "Atrial fibrillation (disorder)":
        "significant but not catastrophic; keeps the composite unambiguously severe",
    "Chronic kidney disease stage 3 (disorder)":
        "not yet organ failure; deliberately retained as a PREDICTOR",
}

SEVERE_CODES = [code for group in SEVERE.values() for code in group]
GROUP_OF = {code: group for group, codes in SEVERE.items() for code in codes}

assert len(SEVERE_CODES) == len(set(SEVERE_CODES)), "duplicate code in SEVERE"
