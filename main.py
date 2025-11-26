import dspy
import random
from tqdm import tqdm
from dspy.teleprompt import BootstrapFewShot

logs = [
    "20171223-22:15:29:606|Step_LSC|30002312|onStandStepChanged 3579",
    "20171223-22:15:29:635|Step_SPUtils|30002312| getTodayTotalDetailSteps = 1514038440000##6993##548365##8661##12266##27164404",
    "20171223-22:15:29:737|Step_LSC|30002312|onExtend:1514038530000 0 0 4",
    "20171223-22:15:29:959|Step_ExtSDM|30002312|calculateCaloriesWithCache totalCalories=126797",
    "20171223-22:15:29:962|Step_StandReportReceiver|30002312|REPORT : 7008 5003 150111 240",
    "20171223-22:15:29:635|Step_StandStepCounter|30002312|flush sensor data",
    "20171223-22:15:29:962|Step_ExtSDM|30002312|calculateAltitudeWithCache totalAltitude=240",
    "20171223-22:15:34:723|Step_StandReportReceiver|30002312|onReceive action: android.intent.action.SCREEN_OFF",
    "20171223-22:15:35:98|Step_SPUtils|30002312|setTodayTotalDetailSteps=1514038440000##7017##548365##8661##13216##27179417",
    "20171223-22:19:58:508|Step_LSC|30002312|flush2DB result success",
]
 
sample_input_string = "\n".join(logs)

try:
    local_llama = dspy.LM(
        model='ollama_chat/llama3.1',
        api_base='http://localhost:11434',
        api_key='none',
        max_tokens=500,
        temperature=0.95
    )
    dspy.settings.configure(lm=local_llama)

    print("Connected to local llama3.1")
except:
    print("could not connect")
    exit()

class GenerateSyntheticLog(dspy.Signature):
    """Analyze the formats of the sample logs then generate synthetic log lines that matches the patterns found in the samples.
    The generated log lines should follow the following instructions:
    1. each log line should start with a timestamp in the format YYYYMMDD-HH:MM:SS:MMM
    2. Each log line should be unique and not a duplicate of any input log line
    3. you must generate new realistic data from the variable parts of the log lines
    4. the parts to change are:
        - timestamp
        - the numeric sections of the events (eg: 7008 5003 150111 240) but keep the same number of sections
    5. keep the non-numeric parts of the log lines the same(eg: onStandStepChanged, REPORT, etc)
    """

    sample_input_log = dspy.InputField(
        desc="A string containing 5-10 log lines separated by newline characters to use as examples for generating new synthetic data"
    )

    synthetic_data = dspy.OutputField(
        description="A single synthetic log line that follows the format of the input logs"
    )

train_example_1 = dspy.Example(
    sample_input_log=sample_input_string,
    synthetic_data="20180312-23:34:122|step_SPUtils|30002345|getTodayTotalDetailSteps = 15815142000##7491##365451##7852##48782##88785214"
).with_inputs('sample_input_log')

train_example_2 = dspy.Example(
    sample_input_log=sample_input_string,
    synthetic_data="20171223-22:16:05:999|Step_StandReportReceiver|30002312|REPORT : 8020 6015 160222 350"
).with_inputs('sample_input_log')

train_example_3 = dspy.Example(
    sample_input_log=sample_input_string,
    synthetic_data="20171223-22:18:45:112|Step_ExtSDM|30002312|calculateCaloriesWithCache totalCalories=129880"
).with_inputs('sample_input_log')

train_set = [train_example_1, train_example_2, train_example_3]

def validate_logic(example, prediction, trace=None):
    
    generated = prediction.synthetic_data
    if not generated:
        return False
    has_length = len(generated) > 30
    has_pipe = "|" in generated
    is_new = generated not in example.sample_input_log
    
    return has_length and has_pipe and is_new

print("compiling model....")

uncompiled_generator = dspy.ChainOfThought(GenerateSyntheticLog)

optimizer = BootstrapFewShot(metric=validate_logic,  max_bootstrapped_demos = 2)

compiled_log_generator = optimizer.compile(student=uncompiled_generator, trainset=train_set)
print("Compilation complete")
# synthetic_logs = []

target_log_count = 50
max_attempts =100

log_set = set()
print("generating logs..")
with tqdm(total=target_log_count, desc="Generating Unique Logs", unit="log") as pbar:
    while len(log_set) < target_log_count and max_attempts >0:
        random_subset = random.sample(logs, 5)
        input_string = "\n".join(random_subset)
        prediction = compiled_log_generator(sample_input_log=input_string)
        synthetic_log = prediction.synthetic_data
        prev_count = len(log_set)
        if synthetic_log:
            log_set.add(synthetic_log)
        if len(log_set) > prev_count:
            pbar.update(1)
        max_attempts -= 1
    
final_logs = list(log_set)
print("generated synthetic logs:\n")
for log in final_logs:
    print(log)

output_path = "outputs/synthetic_logs.txt"
with open(output_path, "w") as f:
    for log in final_logs:
        clean_log = log.strip()
        if clean_log:
            f.write(clean_log+"\n")
print("logs saved")
