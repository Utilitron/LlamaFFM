package ffm.llama.model;

import ffm.llama.binding.LlamaBindings;

import java.lang.foreign.*;
import java.lang.invoke.VarHandle;

/**
 * Wrapper for llama_batch for efficient token processing
 * Supports both prefill (parallel) and decode (sequential) phases
 */
public class LlamaBatch implements AutoCloseable {

    private final MemorySegment batchSegment;
    private final Arena batchArena;
    private final int maxTokens;
    private int nTokens;

    // Pointers to batch data
    private final MemorySegment tokenPtr;
    private final MemorySegment posPtr;
    private final MemorySegment nSeqIdPtr;
    private final MemorySegment seqIdPtr;
    private final MemorySegment logitsPtr;

    /**
     * Create a new batch with specified capacity
     * 
     * @param maxTokens Maximum number of tokens this batch can hold
     * @param maxSeqId Maximum sequence ID (typically 1 for single conversation)
     */
    public LlamaBatch(int maxTokens, int maxSeqId) {
        this.maxTokens = maxTokens;
        this.nTokens = 0;
        this.batchArena = Arena.ofShared();

        try {
            // Initialize batch using llama_batch_init
            this.batchSegment = (MemorySegment) LlamaBindings.llama_batch_init.invoke(batchArena, maxTokens, 0, maxSeqId);

            // Extract pointers from the batch struct for direct manipulation
            VarHandle tokenHandle = LlamaBindings.BATCH_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("token"));
            MemorySegment rawTokenPtr = (MemorySegment) tokenHandle.get(batchSegment, 0L);
            this.tokenPtr = rawTokenPtr.reinterpret((long) maxTokens * Integer.BYTES);

            VarHandle posHandle = LlamaBindings.BATCH_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("pos"));
            MemorySegment rawPosPtr = (MemorySegment) posHandle.get(batchSegment, 0L);
            this.posPtr = rawPosPtr.reinterpret((long) maxTokens * Integer.BYTES);

            VarHandle nSeqIdHandle = LlamaBindings.BATCH_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("n_seq_id"));
            MemorySegment rawNSeqIdPtr = (MemorySegment) nSeqIdHandle.get(batchSegment, 0L);
            this.nSeqIdPtr = rawNSeqIdPtr.reinterpret((long) maxTokens * Integer.BYTES);

            VarHandle seqIdHandle = LlamaBindings.BATCH_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("seq_id"));
            MemorySegment rawSeqIdPtr = (MemorySegment) seqIdHandle.get(batchSegment, 0L);
            this.seqIdPtr = rawSeqIdPtr.reinterpret((long) maxTokens * ValueLayout.ADDRESS.byteSize());

            VarHandle logitsHandle = LlamaBindings.BATCH_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("logits"));
            MemorySegment rawLogitsPtr = (MemorySegment) logitsHandle.get(batchSegment, 0L);
            this.logitsPtr = rawLogitsPtr.reinterpret((long) maxTokens * Byte.BYTES);

            // DO NOT re-allocate seq_id pointers – they are already allocated by llama_batch_init.
            // Just verify they are non-null (they will be).
            for (int i = 0; i < maxTokens; i++) {
                MemorySegment inner = seqIdPtr.getAtIndex(ValueLayout.ADDRESS, i);
                if (inner == MemorySegment.NULL) {
                    throw new IllegalStateException("Internal seq_id pointer is NULL for index " + i);
                }
            }

        } catch (Throwable t) {
            batchArena.close();
            throw new RuntimeException("Failed to initialize batch", t);
        }
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
        LlamaBatch batch = new LlamaBatch(tokens.length, 1);
        
        for (int i = 0; i < tokens.length; i++) {
            // Only compute logits for the last token (in prefill) or as requested
            boolean computeLogits = lastLogits ? (i == tokens.length - 1) : true;
            batch.add(tokens[i], startPos + i, seqId, computeLogits);
        }
        
        return batch;
    }

    /**
     * Add a token to the batch
     * 
     * @param tokenId Token ID
     * @param position Position in sequence
     * @param seqId Sequence ID (0 for single conversation)
     * @param computeLogits Whether to compute logits for this token
     */
    public void add(int tokenId, int position, int seqId, boolean computeLogits) {
        if (nTokens >= maxTokens) {
            throw new IllegalStateException("Batch is full");
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

            // Update n_tokens in the batch struct
            VarHandle nTokensHandle = LlamaBindings.BATCH_LAYOUT.varHandle(
                    MemoryLayout.PathElement.groupElement("n_tokens"));
            nTokensHandle.set(batchSegment, 0L, nTokens);

        } catch (Throwable t) {
            throw new RuntimeException("Failed to add token to batch", t);
        }
    }

    /**
     * Get the native memory segment for this batch
     * Used when calling llama_decode
     */
    public MemorySegment getSegment() {
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
        nTokens = 0;
        try {
            VarHandle nTokensHandle = LlamaBindings.BATCH_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("n_tokens"));
            nTokensHandle.set(batchSegment, 0L, 0);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to clear batch", t);
        }
    }

    @Override
    public void close() {
        try {
            // Free the batch
            LlamaBindings.llama_batch_free.invoke(batchSegment);
        } catch (Throwable t) {
            System.err.println("Warning: Failed to free batch: " + t.getMessage());
        }
    }

    @Override
    public String toString() {
        return String.format("LlamaBatch[size=%d/%d]", nTokens, maxTokens);
    }
}