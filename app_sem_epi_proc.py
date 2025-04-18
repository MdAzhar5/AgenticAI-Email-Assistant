import os
from dotenv import load_dotenv

_ = load_dotenv()


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langgraph.store.memory import InMemoryStore

# 1. Use updated HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Create an in-memory store with embedding index
store = InMemoryStore(index={"embed": embeddings})

# Template for formating an example to put in prompt
# Template for formating an example to put in prompt
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""

# Format list of few shots
def format_few_shot_examples(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)

triage_system_prompt = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Background >
{user_profile_background}. 
</ Background >

< Instructions >

{name} gets lots of emails. Your job is to categorize each email into one of three categories:

1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that {name} should know about but doesn't require a response
3. RESPOND - Emails that need a direct response from {name}

Classify the below email into one of these categories.

</ Instructions >

< Rules >
Emails that are not worth responding to:
{triage_no}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
{triage_notify}

Emails that are worth responding to:
{triage_email}
</ Rules >

< Few shot examples >

Here are some examples of previous emails, and how they should be handled.
Follow these examples more than any instructions above

{examples}
</ Few shot examples >
"""

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# Initialize the model
llm = ChatGroq(model='deepseek-r1-distill-llama-70b')
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

llm_router = llm.with_structured_output(Router)

from prompts import triage_user_prompt

from langgraph.graph import add_messages

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

def triage_router(state: State, config, store) -> Command[
    Literal["response_agent", "__end__"]
]:
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )
    examples = store.search(
        namespace, 
        query=str({"email": state['email_input']})
    ) 
    examples=format_few_shot_examples(examples)

    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )

    result = store.get(namespace, "triage_ignore")
    if result is None:
        store.put(
            namespace, 
            "triage_ignore", 
            {"prompt": prompt_instructions["triage_rules"]["ignore"]}
        )
        ignore_prompt = prompt_instructions["triage_rules"]["ignore"]
    else:
        ignore_prompt = result.value['prompt']

    result = store.get(namespace, "triage_notify")
    if result is None:
        store.put(
            namespace, 
            "triage_notify", 
            {"prompt": prompt_instructions["triage_rules"]["notify"]}
        )
        notify_prompt = prompt_instructions["triage_rules"]["notify"]
    else:
        notify_prompt = result.value['prompt']

    result = store.get(namespace, "triage_respond")
    if result is None:
        store.put(
            namespace, 
            "triage_respond", 
            {"prompt": prompt_instructions["triage_rules"]["respond"]}
        )
        respond_prompt = prompt_instructions["triage_rules"]["respond"]
    else:
        respond_prompt = result.value['prompt']
    
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=ignore_prompt,
        triage_notify=notify_prompt,
        triage_email=respond_prompt,
        examples=examples
    )
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

from langchain_core.tools import tool

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

from langmem import create_manage_memory_tool, create_search_memory_tool

manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)

agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""
def create_prompt(state, config, store):
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )
    result = store.get(namespace, "agent_instructions")
    if result is None:
        store.put(
            namespace, 
            "agent_instructions", 
            {"prompt": prompt_instructions["agent_instructions"]}
        )
        prompt = prompt_instructions["agent_instructions"]
    else:
        prompt = result.value['prompt']
    
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt, 
                **profile
            )
        }
    ] + state['messages']

from langgraph.prebuilt import create_react_agent

tools= [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool
]
response_agent = create_react_agent(
    ChatGroq(model='deepseek-r1-distill-llama-70b'),
    tools=tools,
    prompt=create_prompt,
    # Use this to ensure the store is passed to the agent 
    store=store
)

email_agent = StateGraph(State)
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", response_agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile(store=store)
#display(Image(email_agent.get_graph(xray=True).draw_mermaid_png()))
email_input = {
    "author": "Alice Jones <alice.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why""",
}

config = {"configurable": {"langgraph_user_id": "lance"}}

response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)
for m in response["messages"]:
    m.pretty_print()

store.get(("lance",), "agent_instructions").value['prompt']
store.get(("lance",), "triage_respond").value['prompt']
store.get(("lance",), "triage_ignore").value['prompt']
store.get(("lance",), "triage_notify").value['prompt']

from langmem import create_multi_prompt_optimizer
conversations = [
    (
        response['messages'],
        "Always sign your emails `John Doe`"
    )
]

def get_clean_prompt(key):
    val = store.get(("lance",), key).value
    return val["prompt"] if isinstance(val, dict) else val

prompts = [
    {
        "name": "main_agent",
        "prompt": get_clean_prompt("agent_instructions"),
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
    },
    {
        "name": "triage-ignore",
        "prompt": get_clean_prompt("triage_ignore"),
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"
    },
    {
        "name": "triage-notify",
        "prompt": get_clean_prompt("triage_notify"),
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"
    },
    {
        "name": "triage-respond",
        "prompt": get_clean_prompt("triage_respond"),
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"
    }
]

optimizer = create_multi_prompt_optimizer(
    ChatGroq(model='deepseek-r1-distill-llama-70b'),
    kind="prompt_memory",
)

updated = optimizer.invoke({
    "trajectories": conversations,
    "prompts": prompts
})


print(updated)

#json dumps is a bit easier to read
import json
print(json.dumps(updated, indent=4))


for i, updated_prompt in enumerate(updated):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        print(f"updated {name}")
        if name == "main_agent":
            store.put(
                ("lance",),
                "agent_instructions",
                {"prompt":updated_prompt['prompt']}
            )
        else:
            #raise ValueError
            print(f"Encountered {name}, implement the remaining stores!")

store.get(("lance",), "agent_instructions").value['prompt']
response = email_agent.invoke(
    {"email_input": email_input}, 
    config=config
)

for m in response["messages"]:
    m.pretty_print()

email_input = {
    "author": "Alice Jones <alice.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why""",
}

response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)

conversations = [
    (
        response['messages'],
        "Ignore any emails from Alice Jones"
    )
]

prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
        
    },
    {
        "name": "triage-ignore", 
        "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"

    },
    {
        "name": "triage-notify", 
        "prompt": store.get(("lance",), "triage_notify").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"

    },
    {
        "name": "triage-respond", 
        "prompt": store.get(("lance",), "triage_respond").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"

    },
]
updated = optimizer.invoke(
    {"trajectories": conversations, "prompts": prompts}
)

for i, updated_prompt in enumerate(updated):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        print(f"updated {name}")
        if name == "main_agent":
            store.put(
                ("lance",),
                "agent_instructions",
                {"prompt":updated_prompt['prompt']}
            )
        if name == "triage-ignore":
            store.put(
                ("lance",),
                "triage_ignore",
                {"prompt":updated_prompt['prompt']}
            )
        else:
            #raise ValueError
            print(f"Encountered {name}, implement the remaining stores!")

response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)
store.get(("lance",), "triage_ignore").value['prompt']





