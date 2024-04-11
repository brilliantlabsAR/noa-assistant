from langchain import agents
from langchain import hub
#from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain import memory

questions = [
    "hello what is your name",
    "how are you?",
    "what is your name?",
    "What is 2 multiplied by 3?",
    "how far from the earth is the moon?",
    "wow, that's a lot!",
    "how far is pluto from the earth?",
    "how many times further is than than the moon?"
]

prompt = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions.

Make your responses short (one or two sentences) and precise. Respond without any preamble when giving
translations, just translate directly. When analyzing the user's view, speak as if you can actually
see and never make references to the photo or image you analyzed.

TOOLS:
------

You have access to the following tools:

> Calculator: Useful for when you need to answer questions about math.

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [Calculator]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the user, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

#llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatGroq(model_name="mixtral-8x7b-32768")
tools = agents.load_tools(tool_names=[ "llm-math" ], llm=llm)
memory = memory.ConversationBufferMemory(memory_key="chat_history")
agent = agents.initialize_agent(
    tools=tools,
    llm=llm,
    agent=agents.AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
agent.agent.llm_chain.prompt.template = prompt
print(agent.agent.llm_chain.prompt.template)
for question in questions:
    response = agent.run(
        input=question
    )
print(memory.buffer_as_str)
