# Images from chat

Enable image generation in Settings > Chatbot settings. Ask the chatbot to draw an image; it can pass a visual prompt to the generator using the current Main settings. The chat model is unloaded before image generation to release memory.

Chat bots opens with **Normal** selected. Image generation is a capability of the
conversation, not its default purpose. Greetings, questions, stories and descriptions
receive text responses. Character bots remain available in the selector.

After an image, ask for a change to revise its complete prompt and generate again:
"Update the prompt for Akame to be on the beach in sunglasses in her black fighting
suit." The previous generation prompt stays in chat history; unchanged details
should carry forward. For text without generation, say "Rewrite the prompt only;
do not generate." Change topic normally, for example "Now write a story about her."

**Force image generation** applies to the next message only and resets when sent.
It requires an image-tool call even for a short subject such as "a blue teapot".
This explicit action works even when automatic image-tool selection is disabled
in Settings. The selected model must support function calling. Failed or truncated
tool calls are reported rather than presented as successful images.

Tool selection and prompt quality still depend on the model and its context.
The force option controls the next tool call; it does not guarantee visual identity
or likeness. Keep chat history enabled when revising earlier images.
