package ffm.llama.session.strategy;

import ffm.llama.model.LlamaContext;

/**
 * Sliding window: keep last N% of tokens, discard oldest.
 * <p>
 * This is the strategy currently embedded in LlmService.generateStreaming.
 * Extracting it here makes the behavior explicit and testable.
 */
class SlidingWindowStrategy implements ContextStrategy {
    
    private final int contextSize;
    private final double keepRatio;
    private final int safetyMargin;
    
    SlidingWindowStrategy(int contextSize, double keepRatio, int safetyMargin) {
        if (keepRatio <= 0 || keepRatio > 1) {
            throw new IllegalArgumentException("keepRatio must be in (0, 1]: " + keepRatio);
        }
        if (safetyMargin < 0) {
            throw new IllegalArgumentException("safetyMargin must be non-negative: " + safetyMargin);
        }
        
        this.contextSize = contextSize;
        this.keepRatio = keepRatio;
        this.safetyMargin = safetyMargin;
    }
    
    @Override
    public boolean needsManagement(int currentPosition, LlamaContext context) {
        return currentPosition >= contextSize - safetyMargin;
    }
    
    @Override
    public ManagementAction manage(int currentPosition, int[] cachedTokens, LlamaContext context) {
        int keepTokens = (int) (contextSize * keepRatio);
        
        // Safety check: must keep at least some tokens
        if (keepTokens < 100) {
            keepTokens = Math.min(100, currentPosition);
        }
        
        return ManagementAction.shiftLeft(keepTokens);
    }
}
