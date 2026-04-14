package ffm.llama.message;

//Interface for all messages passed to the LlmService
public interface LlmMessage {
    MessageRole role();
    String content();
}
