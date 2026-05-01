package ffm.llama.session.strategy;

import ffm.llama.context.LlamaContext;

/**
 * No management: let it fail when full.
 * Useful for testing or fixed-size prompts.
 */
class NoManagementStrategy implements ContextStrategy {
    
    @Override
    public boolean needsManagement(int currentPosition, LlamaContext context) {
        return false; // Never manage
    }
    
    @Override
    public ManagementAction manage(int currentPosition, int[] cachedTokens, LlamaContext context) {
        return ManagementAction.none();
    }
}