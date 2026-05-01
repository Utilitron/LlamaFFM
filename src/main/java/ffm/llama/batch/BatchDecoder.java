package ffm.llama.batch;

import ffm.llama.binding.LlamaBindings;
import java.lang.foreign.MemorySegment;

/**
 * Handles batch decoding and retrieval of logits / embeddings.
 */
public class BatchDecoder {
    
    private final MemorySegment ctx;
    
    public BatchDecoder(MemorySegment ctx) {
        this.ctx = ctx;
    }
    
    /**
     * Decode a batch of tokens
     *
     * @param batch The batch to decode
     * @return 0 on success, non-zero on error
     */
    public int decode(LlamaBatch batch) {
        if (this.ctx == null || this.ctx == MemorySegment.NULL) {
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
     * Get embeddings for a specific token in the batch.
     *
     * @param index token index (0‑based)
     */
    public MemorySegment getEmbeddingsIth(int index) {
            try {
            return (MemorySegment) LlamaBindings.llama_get_embeddings_ith.invoke(ctx, index);
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
    
}