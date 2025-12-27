import csv
def normalize_template(text):
    return ''.join(text.split())
generated_templates = {}
try:
    with open ('test/HealthApp_templates_Synthetic.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >1:
                original_template = row[1].strip()
                normalized = normalize_template(original_template)
                generated_templates[normalized] = original_template

except FileNotFoundError:
    print('file not found')
    exit()

reference_templates = {}
try:
    with open('test/HealthApp_templates.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >1:
                template = row[1].strip()
                normalized = normalize_template(template)
                reference_templates[normalized] = template
except FileNotFoundError:
    print('file not found')
    exit()

gen_keys = set(generated_templates.keys())
ref_keys = set(reference_templates.keys())

only_in_generated = gen_keys - ref_keys
only_in_reference = ref_keys - gen_keys

print('\n')
print('templates only in generated set:', len(only_in_generated))
for template in only_in_generated:
    print(generated_templates[template])

print('\n')     
print('templates only in reference set:', len(only_in_reference))
for template in only_in_reference:
    print(reference_templates[template])