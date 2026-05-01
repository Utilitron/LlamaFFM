package ffm.llama.context.state;

import ffm.llama.binding.LlamaBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * State‑save and state‑load operations for a context.
 */
public class ContextStateIO {
    
    private final MemorySegment ctx;
    
    public ContextStateIO(MemorySegment ctx) {
        this.ctx = ctx;
    }
    
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
     * <p>
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
            
            return (long) LlamaBindings.llama_state_seq_save_file.invoke(ctx, pathSeg, seqId, tokens, 0L);
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
            
            return (long) LlamaBindings.llama_state_seq_load_file.invoke(ctx, pathSeg, seqId, tokens, 0L);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to load sequence from file", t);
        }
    }
}
