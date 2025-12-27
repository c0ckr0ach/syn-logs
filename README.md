Running the ollama local server: ollama run llama3.1
API base address: 'http://localhost:11434'


generated data will be stored in the outputs folder. 
Do save your data somewhere else before running the model again as whatever is in the outputs folder gets replaced with new data

model = dspy.LM(
        model='add model name here',
        api_key='add your api key here', 
    )
