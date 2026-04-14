package ffm.llama.message;

// Simple container for the chat message with tools
public record ChatMessageWithTools(
        MessageRole role,
        String content,
        String toolDefinitions
) implements LlmMessageWithTools {}
