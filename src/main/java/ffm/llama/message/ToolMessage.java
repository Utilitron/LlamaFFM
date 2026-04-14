package ffm.llama.message;

// Simple container for the tool message
public record ToolMessage(
        MessageRole role,
        String toolCallId,
        String toolName,
        String content
) implements LlmToolMessage {}
