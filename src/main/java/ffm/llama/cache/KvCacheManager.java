package ffm.llama.cache;

import ffm.llama.binding.LlamaBindings;
import ffm.llama.config.ModelConfig;
import ffm.llama.model.LlamaModel;

import java.lang.foreign.MemorySegment;

/**
 * Manages the KV cache for a {@link LlamaContext}.
 * All methods require the native context pointer and a valid memory handle.
 */
public class KvCacheManager {
    
    private final MemorySegment ctx;
    private final LlamaModel model;
    
    public KvCacheManager(MemorySegment ctx, LlamaModel model) {
        this.ctx = ctx;
        this.model = model;
    }
    
    /**
     * Obtains a handle to the internal KV cache memory.
     * May return NULL if the context hasn't been used yet or the memory
     * subsystem isn't fully initialised.
     */
    private MemorySegment getMemoryHandle() {
        try {
            if (ctx == null || ctx == MemorySegment.NULL) {
                return MemorySegment.NULL;
            }
            return (MemorySegment) LlamaBindings.llama_get_memory.invoke(ctx);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to obtain KV cache memory handle", t);
        }
    }
    
    /**
     * Clear the entire KV cache
     * Use when starting a fresh conversation
     */
    public void clearKvCache() {
        if (this.ctx == null || this.ctx == MemorySegment.NULL) {
            // Log or silently return if the context isn't active
            return;
        }
        try {
            MemorySegment memHandle = getMemoryHandle();
            
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
     * Shifts the context window left by discarding the oldest tokens,
     * keeping the most recent `keepTokens` tokens.
     *
     * @param keepTokens Number of tokens to retain at the end of the sequence
     * @return The number of tokens removed, or 0 if no shift was needed
     */
    public int shiftContextLeft(int keepTokens) {
        int currentTokens = getMaxSequencePosition(0) + 1;
        
        if (currentTokens <= keepTokens) {
            return 0;
        }
        
        int tokensToRemove = currentTokens - keepTokens;
        
        // posEnd is exclusive
        removeKvCacheTokens(0, 0, tokensToRemove);
        
        // shift surviving tokens back to zero
        shiftKvCacheSequence(
                0,
                tokensToRemove,
                currentTokens,
                -tokensToRemove
        );
        
        return tokensToRemove;
    }
    
    /**
     * Remove tokens from a specific sequence in the KV cache
     *
     * @param seqId    Sequence ID (0 for single conversation)
     * @param posStart Start position (inclusive)
     * @param posEnd   End position (exclusive, -1 for all)
     * @return true if successful
     */
    public boolean removeKvCacheTokens(int seqId, int posStart, int posEnd) {
        try {
            MemorySegment memHandle = getMemoryHandle();
            if (memHandle.equals(MemorySegment.NULL)) {
                return false; // context not yet initialized
            }
            return (boolean) LlamaBindings.llama_memory_seq_rm.invoke(memHandle, seqId, posStart, posEnd);
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
            MemorySegment memHandle = getMemoryHandle();
            if (memHandle.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Cannot copy KV cache: context not yet initialized");
            }
            LlamaBindings.llama_memory_seq_cp.invoke(memHandle, seqIdSrc, seqIdDst, posStart, posEnd);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to copy KV cache sequence", t);
        }
    }
    
    /**
     * Keep only a specific sequence in the KV cache, removing all others
     */
    public void keepOnlySequence(int seqId) {
        try {
            MemorySegment memHandle = getMemoryHandle();
            if (memHandle.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Cannot keep sequence: context not yet initialized");
            }
            LlamaBindings.llama_memory_seq_keep.invoke(memHandle, seqId);
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
            MemorySegment memHandle = getMemoryHandle();
            if (memHandle.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Cannot shift KV cache: context not yet initialized");
            }
            LlamaBindings.llama_memory_seq_add.invoke(memHandle, seqId, posStart, posEnd, delta);
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
            MemorySegment memHandle = getMemoryHandle();
            if (memHandle.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Cannot divide KV cache: context not yet initialized");
            }
            LlamaBindings.llama_memory_seq_div.invoke(memHandle, seqId, posStart, posEnd, divisor);
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
            MemorySegment memHandle = getMemoryHandle();
            if (memHandle.equals(MemorySegment.NULL)) {
                return -1;   // context not yet used
            }
            return (int) LlamaBindings.llama_memory_seq_pos_max.invokeExact(memHandle, seqId);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get max sequence position", t);
        }
    }
    
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
     * Print KV information to console
     */
    public void printInfo() {
        System.out.println("Max Seq Pos:     " + getMaxSequencePosition(0));
        System.out.println("Est. KV Cache:   " + String.format("%.2f GB", estimateKvCacheUsageGB()));
        System.out.println("=".repeat(60));
        System.out.println("\n\n");
    }
    
}