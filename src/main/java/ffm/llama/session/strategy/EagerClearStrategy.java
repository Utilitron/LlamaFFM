package ffm.llama.session.strategy;

import ffm.llama.model.LlamaContext;

/**
 * Eager clear: wipe cache when full and start over.
 * Stateless strategy for request/response systems.
 */
class EagerClearStrategy implements ContextStrategy {
    
    private final int contextSize;
    private final int safetyMargin;
    
    EagerClearStrategy(int contextSize, int safetyMargin) {
        this.contextSize = contextSize;
        this.safetyMargin = safetyMargin;
    }
    
    @Override
    public boolean needsManagement(int currentPosition, LlamaContext context) {
        return currentPosition >= contextSize - safetyMargin;
    }
    
    @Override
    public ManagementAction manage(int currentPosition, int[] cachedTokens, LlamaContext context) {
        return ManagementAction.clearCache();
    }
}

