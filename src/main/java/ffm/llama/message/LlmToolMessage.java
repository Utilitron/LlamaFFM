package ffm.llama.message;

//Interface for all messages passed to the LlmService with tools
public interface LlmToolMessage extends LlmMessage {
    String toolCallId();
    String toolName();
}
