import openai
from pydantic import BaseModel
from typing import Optional
import json
import pandas as pd
import inspect
import numpy as np
import io  # Needed to handle string-based CSV
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use Streamlit secrets to get OpenAI API key
api_key = st.secrets["api_key"]

# Securely set your OpenAI API key
if api_key:
    client = openai.OpenAI(api_key=api_key)
    openai.api_key = api_key
else:
    st.warning("Please configure your OpenAI API key in Streamlit secrets to proceed.")

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []

class Response(BaseModel):
    agent: Optional[Agent]
    messages: list

def function_to_schema(func):
    """Converts a Python function into an OpenAI function schema."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {str(e)}")

    parameters = {}
    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")
        parameters[param.name] = {"type": param_type}

    required = [param.name for param in signature.parameters.values() if param.default == inspect._empty]

    return {
        "name": func.__name__,
        "description": (func.__doc__ or "").strip(),
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required,
        }
    }

def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.name  # Use dot notation to access the function name
    args = json.loads(tool_call.arguments)  # Use dot notation to access the arguments
    print(f"{agent_name}: {name}({args})")
    return tools[name](**args)

def perform_calculation(expression: str):
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": None}, {"np": np})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def analyze_spreadsheet(operation: str, data: str):
    """Perform spreadsheet operations using pandas."""
    try:
        # Convert string data to DataFrame
        df = pd.read_csv(io.StringIO(data))  # Using io.StringIO to convert string to file-like object
        
        if operation == "sum":
            result = df.sum()
        elif operation == "mean":
            result = df.mean()
        elif operation == "describe":
            result = df.describe()
        else:
            return f"Unsupported operation: {operation}"
        
        return f"Result of {operation}: {result.to_string()}"
    except Exception as e:
        return f"Error in spreadsheet analysis: {str(e)}"

math_agent = Agent(
    name="Math and Spreadsheet Agent",
    instructions=(
        "You are a math and spreadsheet analysis agent. "
        "When you detect a mathematical expression or spreadsheet-related query, "
        "use the appropriate function to perform calculations or analysis. "
        "For spreadsheets, assume the data is in CSV format."
    ),
    tools=[perform_calculation, analyze_spreadsheet]
)

def run_full_turn(agent, messages):
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}] + messages,
            functions=tool_schemas or None,  # Use 'functions' instead of 'tools'
        )
        message = response.choices[0].message  # Access the message in the response object
        messages.append(message)

        if message.content:
            st.write(f"{current_agent.name}: {message.content}")

        if not message.function_call:
            break

        tool_call = message.function_call
        result = execute_tool_call(tool_call, tools, current_agent.name)

        result_message = {
            "role": "function",
            "name": tool_call.name,  # Use dot notation to access the function name
            "content": result,
        }
        messages.append(result_message)

    # Create a valid instance of the Agent class and pass it to Response
    return Response(agent=current_agent.dict(), messages=messages[num_init_messages:])

# Streamlit app
st.title("Math and Spreadsheet Analysis Agent")

st.write("Welcome! You can ask me to perform calculations or analyze a spreadsheet (in CSV format).")

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataframe Preview:")
        st.dataframe(df)

        operation = st.selectbox("Select an operation to perform on the spreadsheet", ["sum", "mean", "describe"])
        if st.button("Analyze Spreadsheet"):
            result = analyze_spreadsheet(operation, uploaded_file.getvalue().decode("utf-8"))
            st.write(result)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

user_query = st.text_input("Enter a mathematical expression or a query:")
if st.button("Submit Query"):
    if user_query:
        messages.append({"role": "user", "content": user_query})
        response = run_full_turn(math_agent, messages)
        math_agent = Agent(**response.agent)
        messages.extend(response.messages)
if __name__ == "__main__":
    app.run(debug=True, port=5000)
