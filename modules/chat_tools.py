"""Shared image-tool guidance for local chat and API clients."""

IMAGE_TOOL_GUIDANCE = (
    "Respond to the user's current message as a conversational assistant. "
    "Image generation is an optional capability, not your default task. "
    "Use the image tool only when the current user request asks you to create, draw, "
    "edit or regenerate an image. After generating an image, a request to change "
    "its subject, scene, clothing or prompt means revise the previous full prompt "
    "and generate the updated image, preserving details the user did not change. "
    "For example, 'Update the prompt so she is on the beach' after an image "
    "requires a new image-tool call, not just printing an updated prompt. "
    "Answer greetings, questions, stories, descriptions, critique and explicit "
    "prompt-only requests in text. An earlier generation request or "
    "review does not authorize another image. For a generation request, call the "
    "tool with a descriptive visual prompt without asking for extra confirmation. "
    "Never claim an image was generated or edited without calling the tool."
)

IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Create or revise a user-requested image using a complete visual prompt. Not for conversation or prompt-only answers.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": "Visual description of the requested image"}},
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
}
