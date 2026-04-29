package ffm.llama.session.metrics;

/**
 * Performance metrics for a generation session.
 */
public class SessionMetrics {
    
    // Prefill metrics
    private int prefillTokens;
    private long prefillNanos;
    
    // Generation metrics
    private int generationTokens;
    private long generationNanos;
    
    // Context management events
    private int contextShifts;
    private int contextClears;
    
    public SessionMetrics() {
        reset();
    }
    
    public void recordPrefill(int tokens, long nanos) {
        this.prefillTokens += tokens;
        this.prefillNanos += nanos;
    }
    
    public void recordGeneration(int tokens, long nanos) {
        this.generationTokens += tokens;
        this.generationNanos += nanos;
    }
    
    public void recordContextShift() {
        this.contextShifts++;
    }
    
    public void recordContextClear() {
        this.contextClears++;
    }
    
    public void reset() {
        this.prefillTokens = 0;
        this.prefillNanos = 0;
        this.generationTokens = 0;
        this.generationNanos = 0;
        this.contextShifts = 0;
        this.contextClears = 0;
    }
    
    public int getPrefillTokens() {
        return prefillTokens;
    }
    
    public int getGenerationTokens() {
        return generationTokens;
    }
    
    public int getTotalTokens() {
        return prefillTokens + generationTokens;
    }
    
    /**
     * Prefill throughput in tokens per second.
     */
    public double getPrefillTokensPerSecond() {
        if (prefillNanos == 0) return 0;
        return (prefillTokens * 1_000_000_000.0) / prefillNanos;
    }
    
    /**
     * Generation throughput in tokens per second.
     */
    public double getGenerationTokensPerSecond() {
        if (generationNanos == 0) return 0;
        return (generationTokens * 1_000_000_000.0) / generationNanos;
    }
    
    /**
     * Overall throughput in tokens per second.
     */
    public double getTotalTokensPerSecond() {
        long totalNanos = prefillNanos + generationNanos;
        if (totalNanos == 0) return 0;
        return (getTotalTokens() * 1_000_000_000.0) / totalNanos;
    }
    
    /**
     * Prefill latency in milliseconds.
     */
    public double getPrefillLatencyMs() {
        return prefillNanos / 1_000_000.0;
    }
    
    /**
     * Generation latency in milliseconds.
     */
    public double getGenerationLatencyMs() {
        return generationNanos / 1_000_000.0;
    }
    
    /**
     * Total latency in milliseconds.
     */
    public double getTotalLatencyMs() {
        return (prefillNanos + generationNanos) / 1_000_000.0;
    }
    
    public int getContextShifts() {
        return contextShifts;
    }
    
    public int getContextClears() {
        return contextClears;
    }
    
    /**
     * Format metrics as human-readable string.
     */
    public String summary() {
        return String.format(
                "Metrics: prefill=%d tok (%.1f tok/s, %.2fms) | gen=%d tok (%.1f tok/s, %.2fms) | " +
                        "total=%d tok (%.1f tok/s, %.2fms) | shifts=%d clears=%d",
                prefillTokens, getPrefillTokensPerSecond(), getPrefillLatencyMs(),
                generationTokens, getGenerationTokensPerSecond(), getGenerationLatencyMs(),
                getTotalTokens(), getTotalTokensPerSecond(), getTotalLatencyMs(),
                contextShifts, contextClears
        );
    }
    
    @Override
    public String toString() {
        return summary();
    }
}
