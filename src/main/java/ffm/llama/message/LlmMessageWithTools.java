package ffm.llama.message;

//Interface for all messages passed to the LlmService with tools
public interface LlmMessageWithTools extends LlmMessage {
    String toolDefinitions();
}
