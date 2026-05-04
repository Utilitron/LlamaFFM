package ffm.llama.exception;

/**
 * Custom timeout exception when a context cannot be borrowed from
 * the pool within the configured {@code contextBorrowTimeoutMs}.
 */
public class LlmServiceTimeoutException extends RuntimeException {
    public LlmServiceTimeoutException(String message) { super(message); }
}
