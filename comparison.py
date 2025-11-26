import csv

generated_templates = set()
try:
    with open ('outputs/initial_templates.txt', 'r') as f:
        for line in f:
            line = line.strip()
            generated_templates.add(line)

except FileNotFoundError:
    print('file not found')
    exit()

reference_templates = set()
try:
    with open('inputs\HealthApp_2k.log_templates.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >1:
                template = row[1].strip()
                reference_templates.add(template)
except FileNotFoundError:
    print('file not found')
    exit()

only_in_generated = generated_templates - reference_templates
only_in_reference = reference_templates - generated_templates

print('\n')
print('templates only in generated set:', len(only_in_generated))
for template in only_in_generated:
    print(template)

print('\n')     
print('templates only in reference set:', len(only_in_reference))
for template in only_in_reference:
    print(template)