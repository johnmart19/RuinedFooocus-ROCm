"""Shared image-tool guidance for local chat and API clients."""

IMAGE_TOOL_GUIDANCE = (
    "Respond to the user's current message as a conversational assistant. "
    "Image generation is an optional capability, not your default task. "
    "Use the image tool only when the current user request asks you to create, draw, "
    "edit or regenerate an image. A new generation uses the current request; "
    "do not carry subjects, details or prompts from earlier generations into it. "
    "An image reference is used only when the app or client explicitly supplies one. "
    "For an explicit follow-up, preserve the selected reference's unchanged details. "
    "Keep every explicitly requested medium, style, subject attribute, color, count, "
    "spatial relation, exclusion and literal quoted text in the visual prompt. "
    "Do not silently omit or replace them. "
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
        "description": "Generate from the current request. The app controls any explicitly selected image reference. Not for prompt-only answers.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Visual description of the requested image",
                }
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
}
