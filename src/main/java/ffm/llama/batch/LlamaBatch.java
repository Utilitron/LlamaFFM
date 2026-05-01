package ffm.llama.batch;

import ffm.llama.binding.LlamaBindings;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Wrapper for llama_batch for efficient token processing
 * Supports both prefill (parallel) and decode (sequential) phases
 */
public class LlamaBatch implements AutoCloseable {
    
    private final MemorySegment batchSegment;
    private final int maxTokens;

    // Pointers to batch data
    private final MemorySegment tokenPtr;
    private final MemorySegment posPtr;
    private final MemorySegment nSeqIdPtr;
    private final MemorySegment seqIdPtr;
    private final MemorySegment logitsPtr;

    private int nTokens;

    private final Arena batchArena;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Create a new batch with specified capacity
     *
     * @param maxTokens Maximum number of tokens this batch can hold
     * @param maxSeqId  Maximum sequence ID (typically 1 for single conversation)
     */
    public LlamaBatch(int maxTokens, int maxSeqId) {
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be positive, got " + maxTokens);
        }

        if (maxSeqId <= 0) {
            throw new IllegalArgumentException("maxSeqId must be positive, got " + maxSeqId);
        }

        this.maxTokens = maxTokens;
        this.nTokens = 0;
        this.batchArena = Arena.ofConfined();

        try {
            // Batch memory is owned by llama.cpp and must be released with llama_batch_free
            this.batchSegment = (MemorySegment) LlamaBindings.llama_batch_init.invoke(batchArena, maxTokens, 0, maxSeqId);
            
            if (batchSegment == null || batchSegment.address() == 0) {
                throw new IllegalStateException("llama_batch_init returned NULL");
            }
            
            // Extract pointers from the batch struct for direct manipulation using cached VarHandles
            MemorySegment rawTokenPtr = (MemorySegment) LlamaBindings.BATCH_TOKEN.get(batchSegment, 0L);
            this.tokenPtr = reinterpretRequired(rawTokenPtr, (long) maxTokens * Integer.BYTES, "token");

            MemorySegment rawPosPtr = (MemorySegment) LlamaBindings.BATCH_POS.get(batchSegment, 0L);
            this.posPtr = reinterpretRequired(rawPosPtr, (long) maxTokens * Integer.BYTES, "pos");

            MemorySegment rawNSeqIdPtr = (MemorySegment) LlamaBindings.BATCH_N_SEQ_ID.get(batchSegment, 0L);
            this.nSeqIdPtr = reinterpretRequired(rawNSeqIdPtr, (long) maxTokens * Integer.BYTES, "n_seq_id");

            MemorySegment rawSeqIdPtr = (MemorySegment) LlamaBindings.BATCH_SEQ_ID.get(batchSegment, 0L);
            this.seqIdPtr = reinterpretRequired(rawSeqIdPtr, (long) maxTokens * ValueLayout.ADDRESS.byteSize(), "seq_id");

            MemorySegment rawLogitsPtr = (MemorySegment) LlamaBindings.BATCH_LOGITS.get(batchSegment, 0L);
            this.logitsPtr = reinterpretRequired(rawLogitsPtr, (long) maxTokens * Byte.BYTES, "logits");
            
            // Verify seq_id pointers are properly allocated by llama_batch_init
            for (int i = 0; i < maxTokens; i++) {
                MemorySegment inner = seqIdPtr.getAtIndex(ValueLayout.ADDRESS, i);
                if (inner == MemorySegment.NULL) {
                    throw new IllegalStateException("Internal seq_id pointer is NULL for index " + i);
                }
            }
            
        } catch (Throwable t) {
            try { batchArena.close(); } catch (Exception ignored) {}
            throw new RuntimeException("Failed to initialize batch", t);
        }
    }
    
    /**
     * Create a new batch with specified capacity and a default single sequence.
     * Equivalent to {@code new LlamaBatch(maxTokens, 1)}.
     *
     * @param maxTokens Maximum number of tokens this batch can hold
     * @return New LlamaBatch instance
     */
    public static LlamaBatch create(int maxTokens) {
        return new LlamaBatch(maxTokens, 1);
    }
    
    /**
     * Create a batch for a single token (common for decode phase)
     */
    public static LlamaBatch forSingleToken(int tokenId, int position, int seqId) {
        LlamaBatch batch = new LlamaBatch(1, 1);
        batch.add(tokenId, position, seqId, true);
        return batch;
    }
    
    /**
     * Create a batch from an array of tokens (common for prefill phase)
     */
    public static LlamaBatch forTokens(int[] tokens, int startPos, int seqId, boolean lastLogits) {
        if (tokens == null || tokens.length == 0) {
            throw new IllegalArgumentException("tokens must not be null or empty");
        }
        
        LlamaBatch batch = new LlamaBatch(tokens.length, 1);
        
        for (int i = 0; i < tokens.length; i++) {
            // Only compute logits for the last token (in prefill) or as requested
            boolean computeLogits = !lastLogits || (i == tokens.length - 1);
            
            batch.add(tokens[i], startPos + i, seqId, computeLogits);
        }
        
        return batch;
    }
    
    /**
     * Add a token to the batch
     *
     * @param tokenId       Token ID
     * @param position      Position in sequence
     * @param seqId         Sequence ID (0 for single conversation)
     * @param computeLogits Whether to compute logits for this token
     */
    public void add(int tokenId, int position, int seqId, boolean computeLogits) {
        ensureNotClosed();

        if (nTokens >= maxTokens) {
            throw new IllegalStateException("Batch is full");
        }

        if (position < 0) {
            throw new IllegalArgumentException("position must be >= 0, got " + position);
        }

        if (seqId < 0) {
            throw new IllegalArgumentException("seqId must be >= 0, got " + seqId);
        }

        try {
            // Set token
            tokenPtr.setAtIndex(ValueLayout.JAVA_INT, nTokens, tokenId);
            
            // Set position
            posPtr.setAtIndex(ValueLayout.JAVA_INT, nTokens, position);
            
            // Set sequence ID count
            nSeqIdPtr.setAtIndex(ValueLayout.JAVA_INT, nTokens, 1);
            
            // Set sequence ID
            // seq_id is a pointer to array of pointers, so we need to dereference twice
            MemorySegment seqIdArrayPtr = seqIdPtr.getAtIndex(ValueLayout.ADDRESS, nTokens).reinterpret(ValueLayout.JAVA_INT.byteSize());
            seqIdArrayPtr.setAtIndex(ValueLayout.JAVA_INT, 0, seqId);
            
            // Set logits flag
            logitsPtr.setAtIndex(ValueLayout.JAVA_BYTE, nTokens, (byte) (computeLogits ? 1 : 0));
            
            nTokens++;
            
            // Update n_tokens in the batch struct using cached VarHandle
            LlamaBindings.BATCH_N_TOKENS.set(batchSegment, 0L, nTokens);
            
        } catch (Throwable t) {
            throw new RuntimeException("Failed to add token to batch", t);
        }
    }
    
    /**
     * Get the native memory segment for this batch
     * Used when calling llama_decode
     */
    public MemorySegment getSegment() {
        ensureNotClosed();
        return batchSegment;
    }
    
    /**
     * Get current number of tokens in batch
     */
    public int size() {
        return nTokens;
    }
    
    /**
     * Get maximum capacity
     */
    public int capacity() {
        return maxTokens;
    }
    
    /**
     * Check if batch is empty
     */
    public boolean isEmpty() {
        return nTokens == 0;
    }
    
    /**
     * Check if batch is full
     */
    public boolean isFull() {
        return nTokens >= maxTokens;
    }
    
    /**
     * Clear the batch (reset to empty)
     */
    public void clear() {
        ensureNotClosed();

        nTokens = 0;
        try {
            LlamaBindings.BATCH_N_TOKENS.set(batchSegment, 0L, 0);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to clear batch", t);
        }
    }
    
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        
        try {
            // Free the batch memory managed by llama.cpp
            LlamaBindings.llama_batch_free.invoke(batchSegment);
        } catch (Throwable t) {
            System.err.println("Warning: Failed to free batch: " + t.getMessage());
        } finally {
            try { batchArena.close(); } catch (Exception ignored) {}
        }
    }
    
    /**
     * Checks if this batch has been closed.
     *
     * @return true if close() has been called
     */
    public boolean isClosed() {
        return closed.get();
    }
    
    /**
     * Ensures this batch is not closed.
     *
     * @throws IllegalStateException if batch is closed
     */
    private void ensureNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("Batch has been closed");
        }
    }

    private static MemorySegment reinterpretRequired(
            MemorySegment raw,
            long size,
            String fieldName
    ) {
        if (raw == null || raw == MemorySegment.NULL) {
            throw new IllegalStateException(
                    "Native pointer is NULL for field: " + fieldName
            );
        }

        return raw.reinterpret(size);
    }

    @Override
    public String toString() {
        return String.format("LlamaBatch[size=%d/%d, closed=%b]", nTokens, maxTokens, closed.get());
    }
}