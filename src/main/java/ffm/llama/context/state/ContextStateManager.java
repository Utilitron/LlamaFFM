package ffm.llama.context.state;

import ffm.llama.config.ModelConfig;
import ffm.llama.context.LlamaContext;
import ffm.llama.model.LlamaModel;
import ffm.llama.service.LlmService;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Optional;

/**
 * Manages KV cache state snapshots and restoration.
 * Handles the low-level memory operations for saving/loading context state.
 */
public class ContextStateManager {
    
    /**
     * Create a snapshot of the context's KV cache state.
     * Converts native memory to a byte array for safe storage.
     *
     * @param model   the loaded model
     * @param ctx     the context to snapshot
     * @param config  the model configuration
     * @return Optional containing the cached state, or empty if snapshot failed
     */
    public static Optional<CachedContextState> snapshotContext(LlamaModel model, LlamaContext ctx, ModelConfig config) {
        try {
            long stateSize = ctx.contextStateIO().getStateSize();
            if (stateSize <= 0) {
                System.err.println("Invalid state size: " + stateSize + " for model " + model.getPath());
                return Optional.empty();
            }
            
            try (Arena tempArena = Arena.ofConfined()) {
                MemorySegment stateBuffer = tempArena.allocate(stateSize);
                long bytesWritten = ctx.contextStateIO().saveState(stateBuffer, stateSize);
                
                if (bytesWritten != stateSize) {
                    System.err.printf("State size mismatch: expected %d, written %d for model %s%n", stateSize, bytesWritten, model.getPath());
                    return Optional.empty();
                }
                
                // Correct way: extract bytes from MemorySegment
                byte[] stateBytes = stateBuffer.asSlice(0, stateSize).toArray(ValueLayout.JAVA_BYTE);
                
                CachedContextState cachedState = new CachedContextState(
                        stateBytes,
                        config,
                        System.currentTimeMillis(),
                        model.getPath(),
                        ctx.kvCache().getMaxSequencePosition(0) + 1
                );
                
                System.out.printf("Snapshot created for %s: %.2f MB%n", model.getPath(), cachedState.getSizeMB());
                
                return Optional.of(cachedState);
            }
        } catch (Exception e) {
            System.err.println("Failed to snapshot context for " + model.getPath() + ": " + e.getMessage());
            e.printStackTrace();
            return Optional.empty();
        }
    }
    
    /**
     * Restore a cached state into a context.
     * Converts byte array back to native memory and loads it.
     *
     * @param ctx         The context to restore into
     * @param cachedState The cached state to restore
     * @return true if restore was successful, false otherwise
     */
    public static boolean restoreContext(LlamaContext ctx, CachedContextState cachedState) {
        try {
            long stateSize = cachedState.stateBytes().length;
            
            try (Arena tempArena = Arena.ofConfined()) {
                MemorySegment stateBuffer = tempArena.allocate(stateSize);
                
                // Create a source segment from the byte array and copy it
                MemorySegment source = tempArena.allocateFrom(
                        ValueLayout.JAVA_BYTE,
                        cachedState.stateBytes()
                );
                MemorySegment.copy(source, 0, stateBuffer, 0, stateSize);
                
                long bytesRead = ctx.contextStateIO().loadState(stateBuffer, stateSize);
                
                if (bytesRead != stateSize) {
                    System.err.printf("State restore size mismatch: expected %d, read %d", stateSize, bytesRead);
                    return false;
                }
                
                System.out.printf("Restored KV cache for %s (%.2f MB, age: %d ms)", cachedState.modelPath(), cachedState.getSizeMB(), cachedState.getAgeMs());
                
                return true;
            }
        } catch (Exception e) {
            System.err.printf("Failed to restore context: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
}
