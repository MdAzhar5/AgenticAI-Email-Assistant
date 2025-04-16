import os
from dotenv import load_dotenv
_ = load_dotenv()


profile = {
        "name":"Azhar",
        "full_name":"Muhammad Azhar",
        "user_profile_background":"Junior Generative Agentic AI Engineer"

    }

prompt_instructions = {
    "triage_rules": {
        "ignore": "marketing_newsletters, spam_emails, mass_company_announcements",
        "notify": "team_member_out_sick, build_system_notifications, project_status_updates",
        "respond": "direct_questions_from_team, meeting_requests, critical_bug_reports"
    },
    "agent_instructions": "Use appropriate tools to help manage Azhar's meetings, availability, and responses efficiently. Prioritize clarity, timeliness, and relevance in all communications."
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

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from typing import Literal
from pydantic import BaseModel, Field


api_key = os.getenv("GOOGLE_API")

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=api_key)
class Router(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"] = Field(
    description="The classification of an email: 'ignore' for irrelevant emails, "
                    "'notify' for important information that doesn't need a response, "
                    "'respond' for emails that need a reply."
    )

# Step 2: Initialize the Gemini LLM from langchain_google_genai

llm_router = llm.with_structured_output(Router)

from jinja2 import Template

triage_system_prompt_template = Template(
    "Hello {{ full_name }} ({{ name }})!\n"
    "Profile Background: {{ user_profile_background }}\n\n"
    "Triage Rules:\n"
    "  - Ignore: {{ triage_no }}\n"
    "  - Notify: {{ triage_notify }}\n"
    "  - Respond via: {{ triage_email }}\n"
)

triage_user_prompt_template = Template(
    "Email Details:\n"
    "  From: {{ author }}\n"
    "  To: {{ to }}\n"
    "  Subject: {{ subject }}\n\n"
    "Email Thread:\n"
    "{{ email_thread }}"
)

system_prompt = triage_system_prompt_template.render(
    full_name=profile["full_name"],
    name=profile["name"],
    user_profile_background=profile["user_profile_background"],
    triage_no=prompt_instructions["triage_rules"]["ignore"],
    triage_notify=prompt_instructions["triage_rules"]["notify"],
    triage_email=prompt_instructions["triage_rules"]["respond"],
)

user_prompt = triage_user_prompt_template.render(
    author=email["from"],
    to=email["to"],
    subject=email["subject"],
    email_thread=email["body"],
)

result = llm_router.invoke(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
)
print(result)


from langchain_core import tools

@tools
def write_email(to:str, subject : str, content :str) -> str:
    """Write  and Send Email."""
    return f"Email sent {to} with subject '{subject}'"

@tools
def schedule_meeting(
    attendees: list[str],
    subject : str,
    duration_minutes : int,
    preferred_day: str
)-> str:
    
    return f"Meeting'{subject}' scheduled for {preferred_day} with {len(attendees)}attendees"













