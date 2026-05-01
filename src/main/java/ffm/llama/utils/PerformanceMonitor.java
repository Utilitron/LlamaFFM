package ffm.llama.utils;

import ffm.llama.binding.LlamaBindings;

import java.lang.foreign.MemorySegment;

public class PerformanceMonitor {
    
    private final MemorySegment ctx;
    
    public PerformanceMonitor(MemorySegment ctx) {
        this.ctx = ctx;
    }
    
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
}
