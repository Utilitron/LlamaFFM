package ffm.llama.model;

import ffm.llama.binding.LlamaBindings;
import ffm.llama.config.ModelConfig;

import java.lang.foreign.*;
import java.nio.file.Paths;

/**
 * Represents a llama.cpp inference context with model configuration.
 * Manages KV cache, batch processing, and performance monitoring.
 */
public class LlamaContext implements AutoCloseable {

    private final MemorySegment ctx;
    private final LlamaModel model;
    private final ModelConfig modelConfig;
    private final Arena contextArena;

    /**
     * Create a new context with default model configuration
     */
    public LlamaContext(LlamaModel model) {
        this(model, null);
    }

    /**
     * Create a new context with explicit model configuration
     * 
     * @param model The loaded model
     * @param modelConfig Model configuration (null = use model's config)
     */
    public LlamaContext(LlamaModel model, ModelConfig modelConfig) {
        this.model = model;
        this.contextArena = Arena.ofShared();

        // Determine model config to use
        if (modelConfig != null) {
            this.modelConfig = modelConfig;
        } else if (model.getModelConfig() != null) {
            this.modelConfig = model.getModelConfig();
        } else {
            throw new IllegalStateException("Model configuration could not be determined");
        }

        try {
            // Create context parameters struct
            MemorySegment contextParams = contextArena.allocate(LlamaBindings.CONTEXT_PARAMS_LAYOUT);
            
            // Get default parameters
            MemorySegment defaultParams = (MemorySegment) LlamaBindings.llama_context_default_params.invoke(contextArena);
            
            // Copy defaults to our arena
            MemorySegment.copy(defaultParams, 0, contextParams, 0, LlamaBindings.CONTEXT_PARAMS_LAYOUT.byteSize());

            // Apply model configuration
            applyModelConfigToContextParams(contextParams);

            // Create context
            this.ctx = (MemorySegment) LlamaBindings.llama_init_from_model.invoke(
                    model.ptr(), contextParams
            );

            if (ctx == MemorySegment.NULL) {
                contextArena.close();
                throw new RuntimeException("Failed to create context");
            }

        } catch (Throwable t) {
            contextArena.close();
            throw new RuntimeException("Failed to create context", t);
        }
    }

    /**
     * Apply model configuration to context parameters struct
     */
    private void applyModelConfigToContextParams(MemorySegment contextParams) {
        try {
            // Context size
            LlamaBindings.CONTEXT_N_CTX.set(contextParams, 0L, modelConfig.getContextSize());

            // Batch size
            LlamaBindings.CONTEXT_N_BATCH.set(contextParams, 0L, modelConfig.getBatchSize());

            // Physical batch size
            LlamaBindings.CONTEXT_N_UBATCH.set(contextParams, 0L, Math.min(512, modelConfig.getBatchSize()));

            // CPU threads
            LlamaBindings.CONTEXT_N_THREADS.set(contextParams, 0L, modelConfig.getCpuThreads());

            // Batch threads
            LlamaBindings.CONTEXT_N_THREADS_BATCH.set(contextParams, 0L, modelConfig.getCpuThreads());

            // KV cache offloading
            LlamaBindings.CONTEXT_OFFLOAD_KQV.set(contextParams, 0L, (byte) (modelConfig.isOffloadKvToGpu() ? 1 : 0));

            // Flash attention (INT, correct field name)
            LlamaBindings.CONTEXT_FLASH_ATTN_TYPE.set(contextParams, 0L, modelConfig.isFlashAttention() ? 1 : 0);

            // Defragmentation threshold
            LlamaBindings.CONTEXT_DEFRAG_THOLD.set(contextParams, 0L, modelConfig.getDefragThreshold());

            // Performance metrics
            LlamaBindings.CONTEXT_NO_PERF.set(contextParams, 0L, (byte) 0);

            // Embeddings
            LlamaBindings.CONTEXT_EMBEDDINGS.set(contextParams, 0L, (byte) (modelConfig.isEmbeddings() ? 1 : 0));

        } catch (Throwable t) {
            throw new RuntimeException("Failed to apply model config to context params", t);
        }
    }

    /**
     * Get the native pointer to the context
     */
    public MemorySegment ptr() {
        return ctx;
    }

    /**
     * Get the associated model
     */
    public LlamaModel getModel() {
        return model;
    }

    /**
     * Get model configuration
     */
    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    // ============================================================================
    // KV CACHE MANAGEMENT
    // ============================================================================

    /**
     * Clear the entire KV cache
     * Use when starting a fresh conversation
     */
    public void clearKvCache() {
        if (this.ctx == null || this.ctx.address() == 0) {
            // Log or silently return if the context isn't active
            return;
        }
        try {
            MemorySegment memHandle = (MemorySegment) LlamaBindings.llama_get_memory.invoke(this.ctx);

            if (memHandle.equals(MemorySegment.NULL)) {
                return;
            }

            // 'false' resets metadata (standard reset)
            // 'true' wipes the physical buffers.
            LlamaBindings.llama_memory_clear.invoke(memHandle, false);

        } catch (Throwable t) {
            throw new RuntimeException("Failed to clear KV cache", t);
        }
    }

    /**
     * Remove tokens from a specific sequence in the KV cache
     * 
     * @param seqId Sequence ID (0 for single conversation)
     * @param posStart Start position (inclusive)
     * @param posEnd End position (exclusive, -1 for all)
     * @return true if successful
     */
    public boolean removeKvCacheTokens(int seqId, int posStart, int posEnd) {
        try {
            return (boolean) LlamaBindings.llama_memory_seq_rm.invoke(ctx, seqId, posStart, posEnd);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to remove KV cache tokens", t);
        }
    }

    /**
     * Copy KV cache from one sequence to another
     * Useful for branching conversations or speculative decoding
     */
    public void copyKvCacheSequence(int seqIdSrc, int seqIdDst, int posStart, int posEnd) {
        try {
            LlamaBindings.llama_memory_seq_cp.invoke(ctx, seqIdSrc, seqIdDst, posStart, posEnd);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to copy KV cache sequence", t);
        }
    }

    /**
     * Keep only a specific sequence in the KV cache, removing all others
     */
    public void keepOnlySequence(int seqId) {
        try {
            LlamaBindings.llama_memory_seq_keep.invoke(ctx, seqId);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to keep sequence", t);
        }
    }

    /**
     * Add an offset to all positions in a sequence
     * Used for context shifting
     */
    public void shiftKvCacheSequence(int seqId, int posStart, int posEnd, int delta) {
        try {
            LlamaBindings.llama_memory_seq_add.invoke(ctx, seqId, posStart, posEnd, delta);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to shift KV cache sequence", t);
        }
    }

    /**
     * Divide all positions in a sequence by a divisor
     * Used for context compression
     */
    public void divideKvCacheSequence(int seqId, int posStart, int posEnd, int divisor) {
        try {
            LlamaBindings.llama_memory_seq_div.invoke(ctx, seqId, posStart, posEnd, divisor);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to divide KV cache sequence", t);
        }
    }

    /**
     * Get the maximum position in a sequence
     * Returns -1 if sequence is empty
     */
    public int getMaxSequencePosition(int seqId) {
        try {
            return (int) LlamaBindings.llama_memory_seq_pos_max.invokeExact(ctx, seqId);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get max sequence position", t);
        }
    }

    // ============================================================================
    // STATE MANAGEMENT (offloading support)
    // ============================================================================

    /**
     * Get the size of the context state in bytes
     * Used to allocate buffers for state saving
     */
    public long getStateSize() {
        try {
            return (long) LlamaBindings.llama_state_get_size.invoke(ctx);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get state size", t);
        }
    }

    /**
     * Save context state to a memory segment
     * Returns number of bytes written
     * 
     * Used for checkpointing or offloading to SSD
     */
    public long saveState(MemorySegment dst, long dstSize) {
        try {
            return (long) LlamaBindings.llama_state_get_data.invoke(ctx, dst, dstSize);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to save state", t);
        }
    }

    /**
     * Load context state from a memory segment
     * Returns number of bytes read
     */
    public long loadState(MemorySegment src, long srcSize) {
        try {
            return (long) LlamaBindings.llama_state_set_data.invoke(ctx, src, srcSize);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to load state", t);
        }
    }

    /**
     * Save a specific sequence to a file
     * For SSD offloading
     */
    public long saveSequenceToFile(String filePath, int seqId) {
        try (Arena tempArena = Arena.ofConfined()) {
            MemorySegment pathSeg = tempArena.allocateFrom(filePath);
            MemorySegment tokens = MemorySegment.NULL; // Save all tokens
            
            return (long) LlamaBindings.llama_state_seq_save_file.invoke(
                    ctx, pathSeg, seqId, tokens, 0L
            );
        } catch (Throwable t) {
            throw new RuntimeException("Failed to save sequence to file", t);
        }
    }

    /**
     * Load a sequence from a file
     * Used to restore from SSD storage
     */
    public long loadSequenceFromFile(String filePath, int seqId) {
        try (Arena tempArena = Arena.ofConfined()) {
            MemorySegment pathSeg = tempArena.allocateFrom(filePath);
            MemorySegment tokens = MemorySegment.NULL; // Load all tokens
            
            return (long) LlamaBindings.llama_state_seq_load_file.invoke(
                    ctx, pathSeg, seqId, tokens, 0L
            );
        } catch (Throwable t) {
            throw new RuntimeException("Failed to load sequence from file", t);
        }
    }

    // ============================================================================
    // BATCH PROCESSING
    // ============================================================================

    /**
     * Decode a batch of tokens
     * 
     * @param batch The batch to decode
     * @return 0 on success, non-zero on error
     */
    public int decode(LlamaBatch batch) {
        if (this.ctx == null || this.ctx.address() == 0) {
            throw new IllegalStateException("LlamaContext is not initialized or has been closed.");
        }

        try {
            return (int) LlamaBindings.llama_decode.invoke(ctx, batch.getSegment());
        } catch (Throwable t) {
            throw new RuntimeException("Failed to decode batch", t);
        }
    }

    /**
     * Get logits for the last processed token
     * Returns a pointer to float array of size vocab_size
     */
    public MemorySegment getLogits() {
        try {
            return (MemorySegment) LlamaBindings.llama_get_logits.invoke(ctx);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get logits", t);
        }
    }

    /**
     * Get logits for a specific token in the batch
     */
    public MemorySegment getLogitsIth(int i) {
        try {
            return (MemorySegment) LlamaBindings.llama_get_logits_ith.invoke(ctx, i);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get logits", t);
        }
    }

    /**
     * Get embeddings (when context is in embedding mode)
     */
    public MemorySegment getEmbeddings() {
        try {
            return (MemorySegment) LlamaBindings.llama_get_embeddings.invoke(ctx);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get embeddings", t);
        }
    }

    /**
     * Get embeddings (when context is in embedding mode)
     */
    public MemorySegment getEmbeddingsIth(int[] tokens) {
        try {
            return (MemorySegment) LlamaBindings.llama_get_embeddings_ith.invoke(ctx, tokens.length - 1);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get embeddings", t);
        }
    }

    /**
     * Get embeddings for a specific sequence
     */
    public MemorySegment getEmbeddingsSeq(int seqId) {
        try {
            return (MemorySegment) LlamaBindings.llama_get_embeddings_seq.invoke(ctx, seqId);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get embeddings for sequence", t);
        }
    }

    // ============================================================================
    // PERFORMANCE TELEMETRY
    // ============================================================================

    /**
     * Print performance statistics to console
     * Shows token/sec, memory usage, etc.
     */
    public void printPerformanceStats() {
        try {
            LlamaBindings.llama_perf_context_print.invoke(ctx);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to print performance stats", t);
        }
    }

    /**
     * Reset performance counters
     */
    public void resetPerformanceStats() {
        try {
            LlamaBindings.llama_perf_context_reset.invoke(ctx);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to reset performance stats", t);
        }
    }

    // ============================================================================
    // UTILITIES
    // ============================================================================

    /**
     * Estimate current KV cache usage in GB
     */
    public double estimateKvCacheUsageGB() {
        int maxPos = getMaxSequencePosition(0); // Default sequence
        if (maxPos < 0) {
            return 0.0;
        }
        
        return ModelConfig.estimateKvCacheSizeGB(
                maxPos + 1,
                model.getLayerCount(),
                model.getEmbeddingSize()
        );
    }

    /**
     * Print context information to console
     */
    public void printInfo() {
        System.out.println("=".repeat(60));
        System.out.println("Context Information");
        System.out.println("=".repeat(60));
        System.out.println("Context Size:     " + modelConfig.getContextSize());
        System.out.println("Batch Size:       " + modelConfig.getBatchSize());
        System.out.println("CPU Threads:      " + modelConfig.getCpuThreads());
        System.out.println("GPU Layers:       " + modelConfig.getGpuLayers());
        System.out.println("KV on GPU:        " + modelConfig.isOffloadKvToGpu());
        System.out.println("Flash Attn:       " + modelConfig.isFlashAttention());
        //System.out.println("Max Seq Pos:     " + getMaxSequencePosition(0));
        //System.out.println("Est. KV Cache:   " + String.format("%.2f GB", estimateKvCacheUsageGB()));
        System.out.println("=".repeat(60));
        System.out.println("\n\n");
    }

    @Override
    public void close() {
        try {
            // Free the context
            LlamaBindings.llama_free.invoke(ctx);
        } catch (Throwable t) {
            // Log but don't throw - we're in cleanup
            System.err.println("Warning: Failed to free context: " + t.getMessage());
        } finally {
            // Always close the arena
            contextArena.close();
        }
    }
}
