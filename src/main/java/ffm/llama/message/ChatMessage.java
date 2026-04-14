package ffm.llama.message;

// Simple container for the chat message
public record ChatMessage(
        MessageRole role,
        String content
) implements LlmMessage {}
