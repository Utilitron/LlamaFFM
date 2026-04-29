package ffm.llama.exception;

/**
 * Thrown when a generation or prefill operation fails.
 * Wraps low‑level errors (decode failures, native crashes) into a
 * controlled exception that the session layer can propagate.
 */
public class InferenceException extends RuntimeException {
    
    public InferenceException(String message) {
        super(message);
    }
    
    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
}
