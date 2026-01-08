import csv

def normalize_template(text):
    return ''.join(text.split())

generated_templates = {}
reference_templates = {}

try:
    with open('test/HealthApp_templates_Synthetic.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None) 
        for row in reader:
            if len(row) > 1:
                original_template = row[1].strip()
                normalized = normalize_template(original_template)
                generated_templates[normalized] = original_template
except FileNotFoundError:
    print('Error: HealthApp_templates_Synthetic.csv not found')
    exit()

try:
    with open('test/HealthApp_templates.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None) 
        for row in reader:
            if len(row) > 1:
                template = row[1].strip()
                normalized = normalize_template(template)
                reference_templates[normalized] = template
except FileNotFoundError:
    print('Error: HealthApp_templates.csv not found')
    exit()

gen_keys = set(generated_templates.keys())
ref_keys = set(reference_templates.keys())

only_in_generated = gen_keys - ref_keys
only_in_reference = ref_keys - gen_keys
common_keys = gen_keys.intersection(ref_keys) 

output_filename = 'comparison_report.txt'

try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Section 1: Only in Generated
        f.write(f"Templates only in Generated Set: \n")
        for key in sorted(only_in_generated): 
            f.write(f"{generated_templates[key]}\n")
        f.write("\n\n")

        f.write(f" Templates only in Reference Set: \n")
        for key in sorted(only_in_reference):
            f.write(f"{reference_templates[key]}\n")
        f.write("\n\n")

        f.write(f"Common Templates: \n")
        for key in sorted(common_keys):
            f.write(f"{reference_templates[key]}\n")
            
    print(f"complete")

except IOError as e:
    print(f"error writing {e}")