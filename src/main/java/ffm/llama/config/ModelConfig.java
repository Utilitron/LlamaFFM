package ffm.llama.config;

import ffm.llama.enums.KVCacheType;

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

    /** Key cache precision */
    private final KVCacheType cacheTypeK;

    /** Value cache precision */
    private final KVCacheType cacheTypeV;

    /** Optional JSON schema for constrained decoding */
    private final String jsonSchema;

    /** Enable GBNF grammar enforcement */
    private final boolean enableGrammar;

    private final boolean dynamicAttentionSharpening;

    private final double attentionSharpeningFactor;  // c in α(N) = 1 + c × √(ln N)



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
        this.cacheTypeK = builder.cacheTypeK;
        this.cacheTypeV = builder.cacheTypeV;
        this.jsonSchema = builder.jsonSchema;
        this.enableGrammar = builder.enableGrammar;
        this.dynamicAttentionSharpening = builder.dynamicAttentionSharpening;
        this.attentionSharpeningFactor = builder.attentionSharpeningFactor;
    }

    // ============================================================================
    // BUILDER PATTERN
    // ============================================================================

    /**
     * Create a sensible default configuration
     * Designed for consumer-grade hardware with balanced performance
     */
    public static ModelConfig createDefault() {
        return Builder.create()
                .gpuLayers(0)           // CPU-only by default (safest)
                .useMmap(true)          // Enable for large models on SSD
                .useMlock(false)        // Disable to avoid locking RAM
                .contextSize(2048)      // Reasonable default
                .batchSize(512)
                .cpuThreads(Runtime.getRuntime().availableProcessors())
                .flashAttention(false)
                .embeddings(false)
                .cacheTypeK(KVCacheType.F16)    // Standard precision
                .cacheTypeV(KVCacheType.F16)
                .build();
    }

    /**
     * Optimized for long-context RAG on consumer GPUs
     * Uses asymmetric TurboQuant: K=q8_0 (preserve positions), V=tq3_0 (aggressive)
     */
    public static ModelConfig longContextConsumer() {
        return Builder.create()
                .gpuLayers(99)
                .useMmap(true)
                .contextSize(32768)     // 32K context
                .batchSize(512)
                .flashAttention(true)
                .cacheTypeK(KVCacheType.Q8_0)       // Higher precision for keys
                .cacheTypeV(KVCacheType.TQ3_0)      // TurboQuant for values (5.2x compression)
                .dynamicAttentionSharpening(true)   // Mitigate quantization noise
                .attentionSharpeningFactor(0.1)
                .build();
    }

    /**
     * Extreme memory savings for 128K+ contexts
     * Uses full TurboQuant pipeline with QJL error correction
     */
    public static ModelConfig extremeCompression() {
        return Builder.create()
                .gpuLayers(99)
                .useMmap(true)
                .contextSize(131072)    // 128K context
                .batchSize(512)
                .flashAttention(true)
                .cacheTypeK(KVCacheType.TBQP3)      // TurboQuant + QJL (~3.0 bpw)
                .cacheTypeV(KVCacheType.TBQP3)
                .dynamicAttentionSharpening(true)
                .attentionSharpeningFactor(0.15)    // Higher for extreme compression
                .build();
    }

    public static class Builder {
        private int gpuLayers = 0;
        private boolean offloadKvToGpu = false;
        private boolean useMmap = true;
        private boolean useMlock = false;
        private int contextSize = 2048;
        private int batchSize = 512;
        private int cpuThreads = Runtime.getRuntime().availableProcessors();
        private float defragThreshold = 0.1f;
        private boolean flashAttention = false;
        private boolean embeddings = false;
        private KVCacheType cacheTypeK = KVCacheType.F16;
        private KVCacheType cacheTypeV = KVCacheType.F16;
        private String jsonSchema = null;
        private boolean enableGrammar = false;
        private boolean dynamicAttentionSharpening = false;
        private double attentionSharpeningFactor = 0.1;

        private Builder() {}

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

        public Builder cacheTypeK(KVCacheType cacheType) {
            this.cacheTypeK = cacheType;
            return this;
        }

        public Builder cacheTypeV(KVCacheType cacheType) {
            this.cacheTypeV = cacheType;
            return this;
        }

        public Builder cacheType(KVCacheType cacheType) {
            this.cacheTypeK = cacheType;
            this.cacheTypeV = cacheType;
            return this;
        }

        public Builder jsonSchema(String schema) {
            this.jsonSchema = schema;
            this.enableGrammar = (schema != null);
            return this;
        }

        public Builder enableGrammar(boolean enable) {
            this.enableGrammar = enable;
            return this;
        }

        public Builder dynamicAttentionSharpening(boolean enable) {
            this.dynamicAttentionSharpening = enable;
            return this;
        }

        public Builder attentionSharpeningFactor(double factor) {
            if (factor < 0 || factor > 1.0) {
                throw new IllegalArgumentException("Sharpening factor must be between 0 and 1.0");
            }
            this.attentionSharpeningFactor = factor;
            return this;
        }

        public ModelConfig build() {
            return new ModelConfig(this);
        }
    }

    // ============================================================================
    // GETTERS
    // ============================================================================

    public int getGpuLayers() { return gpuLayers;}
    public boolean isOffloadKvToGpu() { return offloadKvToGpu;}
    public boolean isUseMmap() { return useMmap;}
    public boolean isUseMlock() { return useMlock;}
    public int getContextSize() { return contextSize;}
    public int getBatchSize() { return batchSize;}
    public int getCpuThreads() { return cpuThreads;}
    public float getDefragThreshold() { return defragThreshold;}
    public boolean isFlashAttention() { return flashAttention; }
    public boolean isEmbeddings() { return embeddings; }
    public KVCacheType getCacheTypeK() { return cacheTypeK; }
    public KVCacheType getCacheTypeV() { return cacheTypeV; }
    public String getJsonSchema() { return jsonSchema; }
    public boolean isEnableGrammar() { return enableGrammar; }
    public boolean isDynamicAttentionSharpening() { return dynamicAttentionSharpening; }
    public double getAttentionSharpeningFactor() { return attentionSharpeningFactor; }


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
                "ModelConfig[gpu_layers=%d, kv_gpu=%b, ctx=%d, batch=%d, threads=%d, defrag=%.2f, flash=%b, cacheK=%s, cacheV=%s, grammar=%b, dynAttn=%b]",
                gpuLayers, offloadKvToGpu, contextSize, batchSize, cpuThreads, defragThreshold, flashAttention, cacheTypeK.name(), cacheTypeV.name(), enableGrammar, dynamicAttentionSharpening
        );
    }
}
