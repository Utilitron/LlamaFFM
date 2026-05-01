package ffm.llama.context.state;

import ffm.llama.config.ModelConfig;

/**
 * Represents a cached context state (KV cache snapshot) stored in RAM.
 * Created when a model is evicted and used to restore state if the model is reloaded.
 */
public record CachedContextState(
        byte[] stateBytes,
        ModelConfig config,
        long savedAtMs,
        String modelPath,
        int nTokens
) {
    /**
     * Check if this cached state is compatible with a given configuration.
     * Only restore if critical parameters match.
     *
     * @param modelPath The path of the model being loaded
     * @param config    The configuration of the model being loaded
     * @return true if compatible, false otherwise
     */
    public boolean isCompatibleWith(String modelPath, ModelConfig config) {
        // Model path must match exactly
        if (!this.modelPath.equals(modelPath)) {
            return false;
        }
        
        // Critical context parameters must match
        if (this.config.getContextSize() != config.getContextSize()) {
            return false;
        }
        
        if (this.config.getBatchSize() != config.getBatchSize()) {
            return false;
        }
        
        // KV cache types must match (affects memory layout)
        if (this.config.getCacheTypeK() != config.getCacheTypeK()) {
            return false;
        }
        
        if (this.config.getCacheTypeV() != config.getCacheTypeV()) {
            return false;
        }
        
        // Embedding mode must match (affects architecture)
        return this.config.isEmbeddings() == config.isEmbeddings();
    }
    
    /**
     * Get the age of this cached state in milliseconds
     */
    public long getAgeMs() {
        return System.currentTimeMillis() - savedAtMs;
    }
    
    /**
     * Get the size of the cached state in MB
     */
    public double getSizeMB() {
        return stateBytes.length / (1024.0 * 1024.0);
    }
    
    /**
     * Get the number of tokens stored in this state.
     */
    public int getNTokens() {
        return nTokens;
    }
    
    @Override
    public String toString() {
        return String.format("CachedState[path=%s, size=%.2f MB, age=%d ms]",
                modelPath, getSizeMB(), getAgeMs());
    }
}
