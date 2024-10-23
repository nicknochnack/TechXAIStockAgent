from dotenv import load_dotenv
import os
from colorama import init, Fore

# Bring in LLM class
from langchain_ibm import WatsonxLLM

# CrewAI
from crewai_tools import SerperDevTool
from crewai import Crew, Agent, Task

params = {"decoding_method": "greedy", "max_new_tokens": 1000}

llm = WatsonxLLM(
    model_id="mistralai/mistral-large",
    project_id=os.environ["WATSONX_PROJECTID"],
    url="https://us-south.ml.cloud.ibm.com",
    params=params,
)

search = SerperDevTool()
researcher = Agent(
    llm=llm,
    function_calling_llm=llm,
    role="Senior Finance Analyst",
    goal="Research company financials in order to produce insights.",
    backstory="You are a veteran finance analyst with a background in banking and analysis.",
    verbose=True,
    allow_delegation=True,
    tools=[search],
)
# Agent 2
developer = Agent(
    llm=llm,
    function_calling_llm=llm,
    role="Senior Python Engineer",
    goal="Build, Test and Execute code for the research team",
    backstory="You are a veteran Python engineer with a background in computer science and physics.",
    verbose=True,
    allow_delegation=True,
    allow_code_execution=True,
)

if __name__ == "__main__":
    while True:
        description = input(
            Fore.YELLOW + "Enter your prompt here, type \quit to exit: " + Fore.RESET
        )
        expected_output = input(
            Fore.YELLOW + "Whats your desired output: " + Fore.RESET
        )

        if description.lower() == "\quit":
            print(Fore.RED + "Exiting, catch ya later." + Fore.RESET)
            break

        task1 = Task(
            description=description, expected_output=expected_output, agent=researcher
        )
        crew = Crew(
            agents=[researcher, developer],
            tasks=[task1],
            verbose=True,
        )

        # Send to llm
        response = crew.kickoff()
        print(Fore.LIGHTMAGENTA_EX + response + Fore.RESET)
