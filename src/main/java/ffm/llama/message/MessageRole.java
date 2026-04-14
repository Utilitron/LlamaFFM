package ffm.llama.message;

/**
 * Enumeration of message roles in LLM conversations.
 */
public enum MessageRole {
    
    /**
     * System messages - initial instructions or context
     */
    SYSTEM("system"),
    
    /**
     * User messages - input from the human or application
     */
    USER("user"),
    
    /**
     * Assistant messages - responses from the LLM
     */
    ASSISTANT("assistant"),
    
    /**
     * Tool messages - results from function/tool execution
     */
    TOOL("tool");
    
    private final String value;
    
    /**
     * Private constructor.
     * 
     * @param value String representation used in templates
     */
    MessageRole(String value) {
        this.value = value;
    }
    
    /**
     * Get the string value for this role (used in prompt templates).
     * 
     * @return Role string (e.g., "system", "user", "assistant", "tool")
     */
    public String getValue() {
        return value;
    }
    
    /**
     * Parse a string into a MessageRole enum.
     * 
     * @param value Role string (case-insensitive)
     * @return Corresponding MessageRole
     * @throws IllegalArgumentException if the string doesn't match any role
     */
    public static MessageRole fromString(String value) {
        if (value == null) {
            throw new IllegalArgumentException("Role string cannot be null");
        }
        
        for (MessageRole role : values()) {
            if (role.value.equalsIgnoreCase(value)) {
                return role;
            }
        }
        
        throw new IllegalArgumentException(
            "Unknown message role: '" + value + "'. " +
            "Valid roles are: system, user, assistant, tool"
        );
    }
    
    /**
     * Returns the string value for use in templates and logging.
     */
    @Override
    public String toString() {
        return value;
    }
}
