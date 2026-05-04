package ffm.llama.session;

import ffm.llama.batch.BatchFactory;
import ffm.llama.batch.DefaultBatchFactory;
import ffm.llama.batch.LlamaBatch;
import ffm.llama.config.ModelConfig;
import ffm.llama.context.LlamaContext;
import ffm.llama.exception.InferenceException;
import ffm.llama.model.*;
import ffm.llama.context.state.CachedContextState;
import ffm.llama.context.state.ContextStateManager;
import ffm.llama.sampling.LlamaSampler;
import ffm.llama.service.LlmService;
import ffm.llama.session.metrics.SessionMetrics;
import ffm.llama.session.strategy.ContextStrategy;

import java.util.Optional;
import java.util.function.Consumer;

/**
 * High-level orchestration for LLM generation.
 * Composes model, context, sampler, and context strategy into
 * a reusable execution loop.
 */
public class GenerationSession implements AutoCloseable {
    
    private final LlamaModel model;
    private final LlamaContext context;
    private final LlamaSampler sampler;
    private final ContextStrategy contextStrategy;
    private final SessionMetrics metrics;
    private final SessionConfig config;
    private final StateSerializer stateSerializer;
    private final BatchFactory batchFactory;
    
    private int[] cachedTokens;
    private int cachePosition;
    private boolean closed;
    
    private GenerationSession(Builder builder) {
        this.model = builder.model;
        this.context = builder.context;
        this.sampler = builder.sampler;
        this.contextStrategy = builder.contextStrategy;
        this.config = builder.config;
        this.stateSerializer = builder.stateSerializer;
        this.batchFactory = builder.batchFactory;
        this.metrics = new SessionMetrics();
        
        this.cachedTokens = new int[0];
        this.cachePosition = 0;
        this.closed = false;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Prefill the context with initial tokens (system prompt + user input).
     * Uses efficient batched processing.
     *
     * @param tokens Initial tokens to cache
     * @return Number of tokens successfully prefilled
     * @throws IllegalStateException if session is closed or context overflow would occur
     */
    public int prefill(int[] tokens) {
        ensureNotClosed();
        
        if (tokens.length == 0) {
            return 0;
        }
        
        // Check context overflow before attempting prefill
        int contextSize = context.getModelConfig().getContextSize();
        if (cachePosition + tokens.length > contextSize) {
            throw new IllegalStateException(String.format("Prefill would exceed context window: position=%d + tokens=%d > context=%d", cachePosition, tokens.length, contextSize));
        }
        
        long startNanos = System.nanoTime();
        
        try {
            int batchSize = context.getModelConfig().getBatchSize();
            int processed = 0;
            
            // Process in batches for efficiency
            while (processed < tokens.length) {
                int chunkSize = Math.min(batchSize, tokens.length - processed);
                int[] chunk = new int[chunkSize];
                System.arraycopy(tokens, processed, chunk, 0, chunkSize);
                
                boolean isLastChunk = (processed + chunkSize >= tokens.length);
                
                try (LlamaBatch batch = batchFactory.createPrefillBatch(chunk, cachePosition + processed, isLastChunk)) {
                    int result = context.decoder().decode(batch);
                    if (result != 0) {
                        throw new InferenceException("Prefill decode failed with code: " + result);
                    }
                }
                
                processed += chunkSize;
            }
            
            // Update session state
            cachedTokens = tokens.clone();
            cachePosition += processed;
            
            long elapsedNanos = System.nanoTime() - startNanos;
            metrics.recordPrefill(processed, elapsedNanos);
            
            if (config.verbose()) {
                double tokensPerSec = (processed * 1_000_000_000.0) / elapsedNanos;
                System.out.printf("[Session] Prefilled %d tokens in %.2fms (%.1f tok/s)%n", processed, elapsedNanos / 1_000_000.0, tokensPerSec);
            }
            
            return processed;
            
        } catch (Exception e) {
            throw new InferenceException("Prefill failed", e);
        }
    }
    
    /**
     * Generate tokens using the configured sampling strategy.
     * Automatically handles context management via the ContextStrategy.
     *
     * @param callback  Called for each generated token (as string)
     * @param maxTokens Maximum tokens to generate (0 = unlimited, until EOS)
     * @return Number of tokens generated
     */
    public int generate(Consumer<String> callback, int maxTokens) {
        ensureNotClosed();
        
        long startNanos = System.nanoTime();
        int generated = 0;
        
        try {
            while (maxTokens == 0 || generated < maxTokens) {
                // Check if context management is needed
                if (contextStrategy.needsManagement(cachePosition, context)) {
                    handleContextManagement();
                }
                
                // Sample next token
                int nextToken = sampler.sample(context, -1);
                
                // Check for termination
                if (isTerminalToken(nextToken)) {
                    if (config.verbose()) {
                        System.out.printf("[Session] EOS token %d at position %d%n", nextToken, cachePosition);
                    }
                    break;
                }
                
                // Convert token to text and invoke callback
                String tokenText = model.tokenToString(nextToken);
                callback.accept(tokenText);
                
                // Decode token and update KV cache
                try (LlamaBatch batch = batchFactory.createDecodeBatch(nextToken, cachePosition)) {
                    int result = context.decoder().decode(batch);
                    if (result != 0) {
                        throw new InferenceException("Decode failed at position " + cachePosition + " with code: " + result);
                    }
                }
                
                cachePosition++;
                generated++;
            }
            
            long elapsedNanos = System.nanoTime() - startNanos;
            metrics.recordGeneration(generated, elapsedNanos);
            
            if (config.verbose()) {
                double tokensPerSec = (generated * 1_000_000_000.0) / elapsedNanos;
                System.out.printf("[Session] Generated %d tokens in %.2fms (%.1f tok/s)%n", generated, elapsedNanos / 1_000_000.0, tokensPerSec);
            }
            
            return generated;
            
        } catch (Exception e) {
            throw new InferenceException("Generation failed after " + generated + " tokens", e);
        }
    }
    
    /**
     * Generate to a string (convenience method).
     *
     * @param maxTokens Maximum tokens to generate
     * @return Generated text
     */
    public String generateToString(int maxTokens) {
        StringBuilder result = new StringBuilder();
        generate(result::append, maxTokens);
        return result.toString();
    }
    
    /**
     * Clear the KV cache and reset position.
     * Useful for starting a new generation with the same session.
     */
    public void reset() {
        ensureNotClosed();
        context.kvCache().clearKvCache();
        cachedTokens = new int[0];
        cachePosition = 0;
        metrics.reset();
    }
    
    /**
     * Get current cache position.
     */
    public int getCachePosition() {
        return cachePosition;
    }
    
    /**
     * Get session metrics.
     */
    public SessionMetrics getMetrics() {
        return metrics;
    }
    
    /**
     * Take a snapshot of the current KV cache state.
     * Returns null if snapshot fails.
     */
    public CachedContextState snapshot() {
        ensureNotClosed();
        return stateSerializer.snapshot(model, context, context.getModelConfig()).orElse(null);
    }
    
    /**
     * Restore from a cached state.
     *
     * @param state Previously captured state
     * @return true if restore succeeded
     */
    public boolean restore(CachedContextState state) {
        ensureNotClosed();
        
        boolean success = stateSerializer.restoreContext(context, state);
        if (success) {
            // Update session state to match restored context
            this.cachePosition = state.getNTokens();
            this.cachedTokens = new int[0]; // Unknown tokens - state only stores position
            
            if (config.verbose()) {
                System.out.printf("[Session] Restored KV cache: %.2f MB, %d tokens%n",
                        state.getSizeMB(), state.getNTokens());
            }
        }
        
        return success;
    }
    
    @Override
    public void close() {
        if (!closed) {
            if (sampler != null) {
                sampler.close();
            }
            closed = true;
        }
    }
    
    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("Session is closed");
        }
    }
    
    private boolean isTerminalToken(int token) {
        return token == model.getEosToken() || token == model.getEotToken();
    }
    
    private void handleContextManagement() {
        ContextStrategy.ManagementAction action = contextStrategy.manage(cachePosition, cachedTokens, context);
        
        switch (action.type()) {
            case SHIFT_LEFT -> {
                int removed = context.kvCache().shiftContextLeft(action.parameter());
                if (removed > 0) {
                    cachePosition -= removed;
                    // Trim cached tokens array
                    if (cachedTokens.length > action.parameter()) {
                        int[] newCache = new int[action.parameter()];
                        System.arraycopy(cachedTokens, cachedTokens.length - action.parameter(),
                                newCache, 0, action.parameter());
                        cachedTokens = newCache;
                    }
                    
                    if (config.verbose()) {
                        System.out.printf("[Session] Context shifted: removed %d tokens, position now %d%n",
                                removed, cachePosition);
                    }
                }
            }
            case CLEAR_CACHE -> {
                context.kvCache().clearKvCache();
                cachePosition = 0;
                cachedTokens = new int[0];
                
                if (config.verbose()) {
                    System.out.println("[Session] Context cleared");
                }
            }
            case NONE -> { /* nothing */ }
        }
    }
    
    public static class Builder {
        private LlamaModel model;
        private LlamaContext context;
        private LlamaSampler sampler;
        private ContextStrategy contextStrategy;
        private SessionConfig config = SessionConfig.defaults();
        private StateSerializer stateSerializer = new DefaultStateSerializer();
        private BatchFactory batchFactory = new DefaultBatchFactory();
        
        public Builder model(LlamaModel model) {
            this.model = model;
            return this;
        }
        
        public Builder context(LlamaContext context) {
            this.context = context;
            return this;
        }
        
        public Builder sampler(LlamaSampler.SamplerConfig samplerConfig) {
            this.sampler = new LlamaSampler(samplerConfig, model.vocabPtr());
            return this;
        }
        
        public Builder sampler(LlamaSampler sampler) {
            this.sampler = sampler;
            return this;
        }
        
        public Builder contextStrategy(ContextStrategy strategy) {
            this.contextStrategy = strategy;
            return this;
        }
        
        public Builder config(SessionConfig config) {
            this.config = config;
            return this;
        }
        
        public Builder verbose(boolean verbose) {
            this.config = SessionConfig.builder()
                    .verbose(verbose)
                    .build();
            return this;
        }
        
        public Builder stateSerializer(StateSerializer stateSerializer) {
            this.stateSerializer = stateSerializer;
            return this;
        }
        
        public Builder batchFactory(BatchFactory batchFactory) {
            this.batchFactory = batchFactory;
            return this;
        }
        
        public GenerationSession build() {
            if (model == null) {
                throw new IllegalStateException("Model is required");
            }
            if (context == null) {
                throw new IllegalStateException("Context is required");
            }
            if (sampler == null) {
                // Default to greedy sampling
                sampler = new LlamaSampler(LlamaSampler.SamplerConfig.greedy(), model.vocabPtr());
            }
            if (contextStrategy == null) {
                // Default to sliding window with 50% keep ratio
                contextStrategy = ContextStrategy.slidingWindow(
                        context.getModelConfig().getContextSize(),
                        0.5,
                        256
                );
            }
            
            return new GenerationSession(this);
        }
    }
    
    /**
     * Production adapter that delegates to the static ContextStateManager.
     */
    private static class DefaultStateSerializer implements StateSerializer {
        @Override
        public Optional<CachedContextState> snapshot(LlamaModel model, LlamaContext context, ModelConfig config) {
            return ContextStateManager.snapshotContext(model, context, config);
        }
        
        @Override
        public boolean restoreContext(LlamaContext context, CachedContextState state) {
            return ContextStateManager.restoreContext(context, state);
        }
    }
}