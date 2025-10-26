import dspy

try:
    local_llama = dspy.LM(
        model='ollama_chat/llama3.1',
        api_base='http://localhost:11434',
        api_key='none',
        max_tokens=100,
        temperature=0.7
    )
    dspy.settings.configure(lm=local_llama)

    print("Connected to local llama3.1")
except:
    print("could not connect")
    exit()

class generateTemplate(dspy.Signature):
    """Given a log line containing delimiters, identify the event in the fourth column and generate a template for it.
    The template should replace variable parts, especially numbers of the event with placeholders like <*>
    example: 
    log: '...|....|...|onExtend:0000 0000 00 0'
    template: 'onExtend:<*><*><*><*>'
    """

    input_log = dspy.InputField(
        description="A log line containing delimiters and an event in the fourth column."
    )

    output_template = dspy.OutputField(
        description="The gemerated template from the fourth column of the input log"
    )

template_generator = dspy.Predict(generateTemplate)

def processLog(filepath):
    unique_templates = set()
    with open(filepath, 'r')  as file:
        for i,line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            try:
                prediction = template_generator(input_log = line)
                template = prediction.output_template

                unique_templates.add(template)
            except Exception as e:
                print("error processing line {line}")
    
    print("total unique templates: ", len(unique_templates))
    for template in unique_templates:
        print(template)

filePath = 'inputs\healthapp_sample.log'
processLog(filePath)