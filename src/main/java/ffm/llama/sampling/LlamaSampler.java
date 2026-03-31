package ffm.llama.sampling;

import ffm.llama.binding.LlamaBindings;
import ffm.llama.model.LlamaContext;

import java.lang.foreign.*;

/**
 * Sampler for token generation with configurable sampling strategies
 * Supports greedy, top-k, top-p, min-p, and temperature sampling
 */
public class LlamaSampler implements AutoCloseable {

    private final MemorySegment samplerChain;
    private final Arena samplerArena;

    /**
     * Create a sampler with default greedy strategy
     */
    public LlamaSampler() {
        this(SamplerConfig.greedy());
    }

    /**
     * Create a sampler with custom configuration
     */
    public LlamaSampler(SamplerConfig config) {
        this.samplerArena = Arena.ofShared();

        try {
            // Get default sampler chain params
            MemorySegment chainParams = samplerArena.allocate(LlamaBindings.SAMPLER_CHAIN_PARAMS_LAYOUT);
            MemorySegment defaultParams = (MemorySegment) LlamaBindings.llama_sampler_chain_default_params.invoke(samplerArena);

            // Copy returned struct to our arena
            MemorySegment.copy(defaultParams, 0, chainParams, 0, LlamaBindings.SAMPLER_CHAIN_PARAMS_LAYOUT.byteSize());

            // Initialize sampler chain
            this.samplerChain = (MemorySegment) LlamaBindings.llama_sampler_chain_init.invoke(chainParams);

            if (samplerChain == MemorySegment.NULL) {
                samplerArena.close();
                throw new RuntimeException("Failed to initialize sampler chain");
            }

            // Add samplers based on configuration
            buildSamplerChain(config);

        } catch (Throwable t) {
            samplerArena.close();
            throw new RuntimeException("Failed to create sampler", t);
        }
    }

    /**
     * Build the sampler chain based on configuration
     */
    private void buildSamplerChain(SamplerConfig config) throws Throwable {
        // Apply temperature first (if not 1.0)
        if (Math.abs(config.temperature - 1.0f) > 0.01f) {
            MemorySegment tempSampler = (MemorySegment) LlamaBindings.llama_sampler_init_temp.invoke(config.temperature);
            LlamaBindings.llama_sampler_chain_add.invoke(samplerChain, tempSampler);
        }

        // Apply top-k filtering
        if (config.topK > 0) {
            MemorySegment topKSampler = (MemorySegment) LlamaBindings.llama_sampler_init_top_k.invoke(config.topK);
            LlamaBindings.llama_sampler_chain_add.invoke(samplerChain, topKSampler);
        }

        // Apply top-p (nucleus) sampling
        if (config.topP < 1.0f) {
            MemorySegment topPSampler = (MemorySegment) LlamaBindings.llama_sampler_init_top_p.invoke(config.topP, 1L);
            LlamaBindings.llama_sampler_chain_add.invoke(samplerChain, topPSampler);
        }

        // Apply min-p sampling
        if (config.minP > 0.0f) {
            MemorySegment minPSampler = (MemorySegment) LlamaBindings.llama_sampler_init_min_p.invoke(config.minP, 1L);
            LlamaBindings.llama_sampler_chain_add.invoke(samplerChain, minPSampler);
        }

        // Final sampler - greedy or random
        if (config.greedy) {
            MemorySegment greedySampler = (MemorySegment) LlamaBindings.llama_sampler_init_greedy.invoke();
            LlamaBindings.llama_sampler_chain_add.invoke(samplerChain, greedySampler);
        } else {
            MemorySegment distSampler = (MemorySegment) LlamaBindings.llama_sampler_init_dist.invoke(config.seed);
            LlamaBindings.llama_sampler_chain_add.invoke(samplerChain, distSampler);
        }
    }

    /**
     * Sample a token from logits
     * 
     * @param ctx Context containing the logits
     * @param idx Index of the token to sample (-1 for last)
     * @return Sampled token ID
     */
    public int sample(LlamaContext ctx, int idx) {
        try {
            return (int) LlamaBindings.llama_sampler_sample.invoke(samplerChain, ctx.ptr(), idx);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to sample token", t);
        }
    }

    /**
     * Print sampler performance statistics
     */
    public void printPerformanceStats() {
        try {
            LlamaBindings.llama_perf_sampler_print.invoke(samplerChain);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to print sampler stats", t);
        }
    }

    /**
     * Reset sampler performance counters
     */
    public void resetPerformanceStats() {
        try {
            LlamaBindings.llama_perf_sampler_reset.invoke(samplerChain);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to reset sampler stats", t);
        }
    }

    @Override
    public void close() {
        try {
            LlamaBindings.llama_sampler_free.invoke(samplerChain);
        } catch (Throwable t) {
            System.err.println("Warning: Failed to free sampler: " + t.getMessage());
        } finally {
            samplerArena.close();
        }
    }

    /**
     * Configuration for sampling strategies
     */
    public static class SamplerConfig {
        public final float temperature;
        public final int topK;
        public final float topP;
        public final float minP;
        public final boolean greedy;
        public final int seed;

        public SamplerConfig(float temperature, int topK, float topP, float minP, boolean greedy, int seed) {
            this.temperature = temperature;
            this.topK = topK;
            this.topP = topP;
            this.minP = minP;
            this.greedy = greedy;
            this.seed = seed;
        }

        /**
         * Greedy sampling - always pick highest probability token
         * Best for deterministic, focused output
         */
        public static SamplerConfig greedy() {
            return new SamplerConfig(1.0f, 0, 1.0f, 0.0f, true, 0);
        }

        /**
         * Balanced sampling - good for general use
         * Temperature 0.7, top-p 0.9
         */
        public static SamplerConfig balanced() {
            return new SamplerConfig(0.7f, 40, 0.9f, 0.05f, false, (int) System.currentTimeMillis());
        }

        /**
         * Creative sampling - more diverse outputs
         * Temperature 0.9, top-p 0.95
         */
        public static SamplerConfig creative() {
            return new SamplerConfig(0.9f, 0, 0.95f, 0.05f, false, (int) System.currentTimeMillis());
        }

        /**
         * Precise sampling - focused but not deterministic
         * Temperature 0.3, top-k 10
         */
        public static SamplerConfig precise() {
            return new SamplerConfig(0.3f, 10, 0.9f, 0.0f, false, (int) System.currentTimeMillis());
        }

        /**
         * Custom configuration builder
         */
        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private float temperature = 0.7f;
            private int topK = 40;
            private float topP = 0.9f;
            private float minP = 0.05f;
            private boolean greedy = false;
            private int seed = (int) System.currentTimeMillis();

            public Builder temperature(float temperature) {
                this.temperature = temperature;
                return this;
            }

            public Builder topK(int topK) {
                this.topK = topK;
                return this;
            }

            public Builder topP(float topP) {
                this.topP = topP;
                return this;
            }

            public Builder minP(float minP) {
                this.minP = minP;
                return this;
            }

            public Builder greedy(boolean greedy) {
                this.greedy = greedy;
                return this;
            }

            public Builder seed(int seed) {
                this.seed = seed;
                return this;
            }

            public SamplerConfig build() {
                return new SamplerConfig(temperature, topK, topP, minP, greedy, seed);
            }
        }

        @Override
        public String toString() {
            if (greedy) {
                return "SamplerConfig[greedy]";
            }
            return String.format("SamplerConfig[temp=%.2f, topK=%d, topP=%.2f, minP=%.2f]", temperature, topK, topP, minP);
        }
    }
}
