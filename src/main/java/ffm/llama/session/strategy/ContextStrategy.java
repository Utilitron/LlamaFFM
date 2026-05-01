package ffm.llama.session.strategy;

import ffm.llama.context.LlamaContext;

/**
 * Strategy for managing context when approaching the context window limit.
 * <p>
 * This abstraction enables different approaches:
 * - Sliding window (keep last N% of tokens)
 * - Summarization (compress old context)
 * - Retrieval (RAG-style context switching)
 * - Hybrid strategies
 * <p>
 * Separating this from the decode loop makes it easy to:
 * - Test different strategies
 * - Compose strategies
 * - Add new strategies without touching core generation code
 */
public interface ContextStrategy {
    
    /**
     * Sliding window strategy: keeps the most recent tokens when nearing capacity.
     *
     * @param contextSize  Total context window size
     * @param keepRatio    Fraction of context to keep (0.0 - 1.0)
     * @param safetyMargin Tokens to leave free before triggering shift
     * @return SlidingWindowStrategy instance
     */
    static ContextStrategy slidingWindow(int contextSize, double keepRatio, int safetyMargin) {
        return new SlidingWindowStrategy(contextSize, keepRatio, safetyMargin);
    }
    
    /**
     * No-op strategy: never manages context, will fail when full.
     * Useful for testing or when you know your prompt fits.
     */
    static ContextStrategy noManagement() {
        return new NoManagementStrategy();
    }
    
    /**
     * Eager clear strategy: clears cache as soon as it fills up.
     * Useful for stateless request/response scenarios.
     */
    static ContextStrategy eagerClear(int contextSize, int safetyMargin) {
        return new EagerClearStrategy(contextSize, safetyMargin);
    }
    
    /**
     * Check if context management is needed at the current position.
     *
     * @param currentPosition Current token position in context
     * @param context         The LlamaContext instance
     * @return true if management action should be taken
     */
    boolean needsManagement(int currentPosition, LlamaContext context);
    
    // ========== Factory Methods ==========
    
    /**
     * Perform context management and return the action taken.
     *
     * @param currentPosition Current position in context
     * @param cachedTokens    Tokens currently in KV cache (may be empty if unknown)
     * @param context         The LlamaContext instance
     * @return Action that was performed
     */
    ManagementAction manage(int currentPosition, int[] cachedTokens, LlamaContext context);
    
    enum ActionType {
        NONE,           // No action needed
        SHIFT_LEFT,     // Remove old tokens, keep recent ones
        CLEAR_CACHE     // Clear entire cache and restart
    }
    
    /**
     * Action taken by a context management strategy.
     */
    record ManagementAction(ActionType type, int parameter) {
        
        public static ManagementAction none() {
            return new ManagementAction(ActionType.NONE, 0);
        }
        
        public static ManagementAction shiftLeft(int keepTokens) {
            return new ManagementAction(ActionType.SHIFT_LEFT, keepTokens);
        }
        
        public static ManagementAction clearCache() {
            return new ManagementAction(ActionType.CLEAR_CACHE, 0);
        }
    }
}