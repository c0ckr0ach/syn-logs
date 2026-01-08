import google.genai as genai
import os

# 1. Setup your key
# Replace with your actual key or read from file
os.environ["GEMINI_API_KEY"] = "AIzaSyCiK1Fxe2t7PHVO07OsFfr3TPxPytnpLCM" 
client = genai.Client()

print("Checking available models...\n")

try:
    for model in client.models.list():
        # In the new SDK, we check 'supported_actions'
        # Common actions: 'generateContent', 'embedContent', 'countTokens'
        actions = getattr(model, 'supported_actions', [])
        
        print(f"🔹 ID: {model.name}")
        print(f"   Name: {model.display_name}")
        if actions:
            print(f"   Actions: {', '.join(actions)}")
        print("-" * 30)
            
except Exception as e:
    print(f"❌ Error: {e}")
    # If the above fails, let's just print the raw names
    print("\nFalling back to raw name list:")
    for model in client.models.list():
        print(f"👉 {model.name}")