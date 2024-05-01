#
# context.py
#
# Routines for creating a message containing additional context about the user. These messages
# should be injected into the conversation.
#
# Information about the user can be extracted by analyzing batches of their messages and turned into
# a simple list of key-value pairs. Feeding these back to the assistant will produce more relevant,
# contextually-aware, and personalized responses.
#

from typing import Dict, List

import openai
import groq

from models import Role, Message, TokenUsage, accumulate_token_usage


####################################################################################################
# Prompts
####################################################################################################

# These are context keys we try to detect in conversation history over time
LEARNED_CONTEXT_KEY_DESCRIPTIONS = {
    "UserName": "User's name",
    "DOB": "User's date of birth",
    "Food": "Foods and drinks user has expressed interest in"
}

LEARNED_CONTEXT_EXTRACTION_SYSTEM_MESSAGE = f"""
Given a transcript of what the user said, look for any of the following information being revealed:

""" + "\n".join([ key + ": "  + description for key, description in LEARNED_CONTEXT_KEY_DESCRIPTIONS.items() ]) + """

Make sure to list them in this format:

KEY=VALUE

If nothing was found, just say "END". ONLY PRODUCE ITEMS WHEN THE USER HAS ACTUALLY REVEALED THEM.
"""

CONTEXT_SYSTEM_MESSAGE_PREFIX = "## Additional context about the user:"


####################################################################################################
# Functions
####################################################################################################

def create_context_system_message(local_time: str | None, location: str | None, learned_context: Dict[str,str] | None) -> str:
    """
    Creates a string of additional context that can either be appended to the main system
    message or as a secondary system message before delivering the assistant response. This is
    how GPT is made aware of the user's location, local time, and any learned information that
    was extracted from prior conversation.

    Parameters
    ----------
    local_time : str | None
        Local time, if known.
    location : str | None
        Location, as a human readable address, if known.
    learned_context : Dict[str,str] | None
        Information learned from prior conversation as key-value pairs, if any.

    Returns
    -------
    str
        Message to combine with existing system message or to inject as a new, extra system
        message.
    """
    # Fixed context: things we know and need not extract from user conversation history
    context: Dict[str, str] = {}
    if local_time is not None and len(local_time) > 0:
        context["current_time"] = local_time
    else:
        context["current_time"] = "If asked, tell user you don't know current date or time because clock is broken"
    if location is not None and len(location) > 0:
        context["location"] = location
    else:
        context["location"] = "You do not know user's location and if asked, tell them so"

    # Merge in learned context
    if learned_context is not None:
        context.update(learned_context)

    # Convert to a list to be appended to a system message or treated as a new system message
    system_message_fragment = CONTEXT_SYSTEM_MESSAGE_PREFIX + "\n".join([ f"<{key}>{value}</{key}>" for key, value in context.items() if value is not None ])
    return system_message_fragment

async def extract_learned_context(
    client: openai.AsyncOpenAI | groq.AsyncGroq,
    message_history: List[Message],
    model: str,
    existing_learned_context: Dict[str, str],
    token_usage_by_model: Dict[str, TokenUsage]
) -> Dict[str, str]:
    # Grab last N user messages
    max_user_history = 2
    messages: List[Message] = []
    for i in range(len(message_history) - 1, -1, -1):
        if len(messages) >= max_user_history:
            break
        if message_history[i].role == Role.USER:
            messages.append(message_history[i])

    # Insert system message and reverse so that it is in the right order
    messages.append(Message(role=Role.SYSTEM, content=LEARNED_CONTEXT_EXTRACTION_SYSTEM_MESSAGE))
    messages.reverse()

    # print("Context extraction input:")
    # print(messages)

    # Process
    response = await client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Do not forget to count tokens used!
    accumulate_token_usage(
        token_usage_by_model=token_usage_by_model,
        model=model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens
    )

    # # Debug: print raw output of context extraction
    # print("Learned context:")
    # print(response.choices[0].message.content)

    # Parse it into a dictionary
    learned_context: Dict[str,str] = {}
    lines = response.choices[0].message.content.splitlines()
    for line in lines:
        parts = line.split("=")
        if len(parts) == 2:
            key, value = parts
            if key in LEARNED_CONTEXT_KEY_DESCRIPTIONS:
                learned_context[key] = value
    
    # Merge with existing
    existing_learned_context.update(learned_context)
    return existing_learned_context