package ffm.llama.config;

/**
 * Configuration for LLM Model
 */
public class ModelConfig {

    // ============================================================================
    // CONFIGURATION FIELDS
    // ============================================================================

    /** Number of layers to offload to GPU (Tier 0) */
    private final int gpuLayers;

    /** Whether to offload KV cache to GPU */
    private final boolean offloadKvToGpu;

    /** Use memory mapping for model loading (critical for Tier 2) */
    private final boolean useMmap;

    /** Lock model in RAM to prevent swap to disk */
    private final boolean useMlock;

    /** Maximum context size in tokens */
    private final int contextSize;

    /** Batch size for prompt processing (prefill phase) */
    private final int batchSize;

    /** Number of CPU threads for offloaded layers */
    private final int cpuThreads;

    /** Defragmentation threshold for KV cache (0.0 = disabled, 0.1 = aggressive) */
    private final float defragThreshold;

    /** Enable flash attention (reduces memory, increases compute) */
    private final boolean flashAttention;

    /** Enable embeddings */
    private final boolean embeddings;

    // ============================================================================
    // CONSTRUCTORS
    // ============================================================================

    private ModelConfig(Builder builder) {
        this.gpuLayers = builder.gpuLayers;
        this.offloadKvToGpu = builder.offloadKvToGpu;
        this.useMmap = builder.useMmap;
        this.useMlock = builder.useMlock;
        this.contextSize = builder.contextSize;
        this.batchSize = builder.batchSize;
        this.cpuThreads = builder.cpuThreads;
        this.defragThreshold = builder.defragThreshold;
        this.flashAttention = builder.flashAttention;
        this.embeddings = builder.embeddings;
    }

    // ============================================================================
    // BUILDER PATTERN
    // ============================================================================

    public static class Builder {
        private int gpuLayers = 0;
        private boolean offloadKvToGpu = false;
        private boolean useMmap = false;
        private boolean useMlock = false;
        private int contextSize = 2048;
        private int batchSize = 512;
        private int cpuThreads = 4;
        private float defragThreshold = 0.1f;
        private boolean flashAttention = false;
        private boolean embeddings = false;

        public static Builder create() {
            return new Builder();
        }

        public Builder gpuLayers(int gpuLayers) {
            this.gpuLayers = gpuLayers;
            return this;
        }

        public Builder offloadKvToGpu(boolean offloadKvToGpu) {
            this.offloadKvToGpu = offloadKvToGpu;
            return this;
        }

        public Builder useMmap(boolean useMmap) {
            this.useMmap = useMmap;
            return this;
        }

        public Builder useMlock(boolean useMlock) {
            this.useMlock = useMlock;
            return this;
        }

        public Builder contextSize(int contextSize) {
            this.contextSize = contextSize;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder cpuThreads(int cpuThreads) {
            this.cpuThreads = cpuThreads;
            return this;
        }

        public Builder defragThreshold(float defragThreshold) {
            this.defragThreshold = defragThreshold;
            return this;
        }

        public Builder flashAttention(boolean flashAttention) {
            this.flashAttention = flashAttention;
            return this;
        }

        public Builder embeddings(boolean embeddings) {
            this.embeddings = embeddings;
            return this;
        }

        public ModelConfig build() {
            return new ModelConfig(this);
        }
    }

    // ============================================================================
    // GETTERS
    // ============================================================================

    public int getGpuLayers() {
        return gpuLayers;
    }

    public boolean isOffloadKvToGpu() {
        return offloadKvToGpu;
    }

    public boolean isUseMmap() {
        return useMmap;
    }

    public boolean isUseMlock() {
        return useMlock;
    }

    public int getContextSize() {
        return contextSize;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getCpuThreads() {
        return cpuThreads;
    }

    public float getDefragThreshold() {
        return defragThreshold;
    }

    public boolean isFlashAttention() {
        return flashAttention;
    }

    public boolean isEmbeddings() { return embeddings; }

    // ============================================================================
    // UTILITIES
    // ============================================================================

    /**
     * Calculate estimated KV cache size in GB for given context
     * Formula: n_ctx * n_layers * n_embd * 2 (K+V) * bytes_per_element
     * Assumes FP16 KV cache (2 bytes per element)
     */
    public static double estimateKvCacheSizeGB(int nCtx, int nLayers, int nEmbd) {
        long totalElements = (long) nCtx * nLayers * nEmbd * 2; // K + V
        long totalBytes = totalElements * 2; // FP16 = 2 bytes
        return totalBytes / 1_000_000_000.0;
    }

    @Override
    public String toString() {
        return String.format(
                "ModelConfig[gpu_layers=%d, kv_gpu=%b, ctx=%d, batch=%d, threads=%d, defrag=%.2f, flash=%b, embeddings=%b]",
                gpuLayers, offloadKvToGpu, contextSize, batchSize, cpuThreads, defragThreshold, flashAttention, embeddings
        );
    }
}
