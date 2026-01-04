import dspy
import random
import os
import time
import litellm
from tqdm import tqdm


os.environ["GROQ_API_KEY"] = "<API_KEY>"

try:
    model = dspy.LM(
        model='groq/llama-3.1-8b-instant', 
        max_tokens=1000,
        temperature=0.95 
    )
    dspy.settings.configure(lm=model)
    print("Connected to model")
except Exception as e:
    print(f"Could not connect: {e}")
    exit()

litellm.drop_params = True

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

class GenerateSyntheticLog(dspy.Signature):
    """Analyze the provided log examples. Generate ONE new synthetic log line that follows the exact same format.
    - Start with a timestamp: YYYYMMDD-HH:MM:SS:MMM
    - Maintain the pipe '|' separators.
    - Create realistic variations of the numbers.
    - Do NOT copy the input examples. Create new data.
    - OUTPUT ONLY THE RAW LOG LINE. NO MARKDOWN."""
    
    sample_input_log = dspy.InputField(desc="Examples of log lines")
    synthetic_data = dspy.OutputField(desc="A single generated log line")

generator = dspy.Predict(GenerateSyntheticLog)


target_log_count = 500
log_set = set()

for log in logs:
    log_set.add(log.strip())

initial_count = len(log_set)
DELAY_SECONDS = 2

print(f"Generating {target_log_count} NEW logs (starting with {initial_count} examples)...")

with tqdm(total=target_log_count, unit="log") as pbar:
    while (len(log_set) - initial_count) < target_log_count:
        
        random_subset = random.sample(logs, 3)
        input_string = "\n".join(random_subset)
        
        try:
            prediction = generator(sample_input_log=input_string)
            raw_log = prediction.synthetic_data
            
            clean_log = raw_log.replace("```", "").strip()
            if "\n" in clean_log: clean_log = clean_log.split("\n")[0]
            
            if clean_log and "|" in clean_log and len(clean_log) > 20:

                if clean_log not in log_set:
                    log_set.add(clean_log)
                    pbar.update(1)
                    
        except Exception as e:
            if "429" in str(e):
                print("Rate limit hit, sleeping...")
                time.sleep(5)
            else:
                print(f"Error: {e}")
        
        time.sleep(DELAY_SECONDS)


final_logs = [log for log in log_set if log not in logs]

print(f"\nGenerated {len(final_logs)} NEW unique logs.")

output_path = "outputs/synthetic_logs.txt"
os.makedirs("outputs", exist_ok=True)

with open(output_path, "w") as f:
    for log in final_logs:
        f.write(log + "\n")
print("Logs saved.")