package ffm.llama.service;

import ffm.llama.binding.LlamaBindings;
import ffm.llama.config.ModelConfig;
import ffm.llama.model.LlamaBatch;
import ffm.llama.model.LlamaContext;
import ffm.llama.model.LlamaModel;
import ffm.llama.sampling.LlamaSampler;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

/**
 * High-level LLM service
 */
public class LlmService implements AutoCloseable {

    // Initialize llama.cpp backend once
    static {
        LlamaBindings.init();
    }

    // Model registry - maps model paths to loaded instances
    private final Map<String, ModelInstance> loadedModels = new ConcurrentHashMap<>();

    /**
     * Model instance wrapper
     */
    private static class ModelInstance {
        final LlamaModel model;
        final LlamaContext context;
        final ModelConfig modelConfig;
        long lastUsedMs;

        ModelInstance(LlamaModel model, LlamaContext context, ModelConfig modelConfig) {
            this.model = model;
            this.context = context;
            this.modelConfig = modelConfig;
            this.lastUsedMs = System.currentTimeMillis();
        }

        void updateLastUsed() {
            this.lastUsedMs = System.currentTimeMillis();
        }

        void close() {
            context.close();
            model.close();
        }
    }

    /**
     * Load a model with default configuration
     * 
     * @param modelPath Path to .gguf model file
     * @return Model identifier for subsequent calls
     */
    public String loadModel(String modelPath) {
        return loadModel(modelPath, null);
    }

    /**
     * Load a model with explicit model configuration
     * 
     * @param modelPath Path to .gguf model file
     * @param modelConfig Model configuration (null = default settings)
     * @return Model identifier for subsequent calls
     */
    public String loadModel(String modelPath, ModelConfig modelConfig) {
        // Check if already loaded
        if (loadedModels.containsKey(modelPath)) {
            return modelPath;
        }

        try {
            // Load model
            LlamaModel model = new LlamaModel(modelPath, modelConfig);
            
            // default config if not provided
            ModelConfig finalConfig = modelConfig;
            if (finalConfig == null) {
                finalConfig = ModelConfig.Builder.create()
                        .gpuLayers(model.getLayerCount())
                        .offloadKvToGpu(true)
                        .contextSize(model.getTrainContextSize())
                        .batchSize(512)
                        .cpuThreads(4)
                        .defragThreshold(0.1f)
                        .flashAttention(true)
                        .build();
            }

            // Create context
            LlamaContext context = new LlamaContext(model, finalConfig);

            // Register instance
            ModelInstance instance = new ModelInstance(model, context, finalConfig);
            loadedModels.put(modelPath, instance);

            System.out.println("Loaded model: " + modelPath);
            model.printInfo();
            context.printInfo();

            return modelPath;

        } catch (Exception e) {
            throw new RuntimeException("Failed to load model: " + modelPath, e);
        }
    }

    /**
     * Unload a model to free VRAM
     */
    public void unloadModel(String modelPath) {
        ModelInstance instance = loadedModels.remove(modelPath);
        if (instance != null) {
            instance.close();
            System.out.println("Unloaded model: " + modelPath);
        }
    }

    /**
     * Generate text with greedy sampling
     * 
     * @param modelPath Model identifier
     * @param prompt Input text
     * @param maxTokens Maximum tokens to generate
     * @return Generated text
     */
    public String generate(String modelPath, String prompt, int maxTokens) {
        return generate(modelPath, prompt, maxTokens, LlamaSampler.SamplerConfig.greedy());
    }

    /**
     * Generate text with custom sampling configuration
     * 
     * @param modelPath Model identifier
     * @param prompt Input text
     * @param maxTokens Maximum tokens to generate
     * @param samplerConfig Sampling strategy
     * @return Generated text
     */
    public String generate(String modelPath, String prompt, int maxTokens, LlamaSampler.SamplerConfig samplerConfig) {
        StringBuilder result = new StringBuilder();
        generateStreaming(modelPath, prompt, maxTokens, samplerConfig, result::append);
        return result.toString();
    }

    /**
     * Generate text with streaming callback
     * Calls the callback for each generated token
     * 
     * @param modelPath Model identifier
     * @param prompt Input text
     * @param maxTokens Maximum tokens to generate
     * @param samplerConfig Sampling strategy
     * @param callback Function called with each generated token
     */
    public void generateStreaming(
            String modelPath,
            String prompt,
            int maxTokens,
            LlamaSampler.SamplerConfig samplerConfig,
            Consumer<String> callback
    ) {
        ModelInstance instance = loadedModels.get(modelPath);
        if (instance == null) {
            throw new IllegalArgumentException("Model not loaded: " + modelPath);
        }

        instance.updateLastUsed();
        LlamaModel model = instance.model;
        LlamaContext ctx = instance.context;

        try (LlamaSampler sampler = new LlamaSampler(samplerConfig)) {
            
            // Clear KV cache for fresh generation
            ctx.clearKvCache();

            // Tokenize prompt
            int[] promptTokens = model.tokenize(prompt, true, true);
            
            // Prefill phase - process prompt in parallel
            try (LlamaBatch batch = LlamaBatch.forTokens(promptTokens, 0, 0, true)) {
                int ret = ctx.decode(batch);
                if (ret != 0) {
                    throw new RuntimeException("Failed to decode prompt batch");
                }
            }

            // Decode phase - generate tokens one by one
            int nCur = promptTokens.length;
            int nDecoded = 0;

            while (nDecoded < maxTokens) {
                // Sample next token
                int nextToken = sampler.sample(ctx, -1);

                // Check for EOS
                if (nextToken == model.getEosToken() || nextToken == model.getEotToken()) {
                    break;
                }

                // Convert to text and callback
                String tokenText = model.tokenToString(nextToken);
                callback.accept(tokenText);

                // Decode next position
                try (LlamaBatch batch = LlamaBatch.forSingleToken(nextToken, nCur, 0)) {
                    int ret = ctx.decode(batch);
                    if (ret != 0) {
                        throw new RuntimeException("Failed to decode token");
                    }
                }

                nCur++;
                nDecoded++;
            }

        } catch (Exception e) {
            throw new RuntimeException("Generation failed for model: " + modelPath, e);
        }
    }

    /**
     * Generate embeddings for text
     * Requires model loaded with embeddings=true in context params
     * 
     * @param modelPath Model identifier (should be embedding model like nomic-embed)
     * @param text Text to embed
     * @return Embedding vector as float array
     */
    public float[] embed(String modelPath, String text) {
        ModelInstance instance = loadedModels.get(modelPath);
        if (instance == null) {
            throw new IllegalArgumentException("Model not loaded: " + modelPath);
        }

        instance.updateLastUsed();
        LlamaModel model = instance.model;
        LlamaContext ctx = instance.context;

        try {
            // Tokenize text
            int[] tokens = model.tokenize(text, true, false);

            // Process batch
            try (LlamaBatch batch = LlamaBatch.forTokens(tokens, 0, 0, false)) {
                int ret = ctx.decode(batch);
                if (ret != 0) {
                    throw new RuntimeException("Failed to process batch for embeddings");
                }
            }

            // Get embeddings
            var embeddingsSeg = ctx.getEmbeddings();
            if (embeddingsSeg == null || embeddingsSeg == MemorySegment.NULL) {
                throw new RuntimeException("No embeddings returned (ensure model is in embedding mode)");
            }

            // Copy to Java array
            int embdSize = model.getEmbeddingSize();
            float[] embeddings = new float[embdSize];
            for (int i = 0; i < embdSize; i++) {
                embeddings[i] = embeddingsSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            }

            return embeddings;

        } catch (Exception e) {
            throw new RuntimeException("Embedding generation failed", e);
        }
    }

    /**
     * Get information about a loaded model
     */
    public ModelInfo getModelInfo(String modelPath) {
        ModelInstance instance = loadedModels.get(modelPath);
        if (instance == null) {
            return null;
        }

        LlamaModel model = instance.model;
        return new ModelInfo(
                modelPath,
                model.getParameterCount(),
                model.getModelSizeGB(),
                instance.modelConfig,
                instance.lastUsedMs
        );
    }

    /**
     * Get all loaded models
     */
    public List<ModelInfo> getLoadedModels() {
        return loadedModels.values().stream()
                .map(inst -> getModelInfo(inst.model.getPath()))
                .toList();
    }

    /**
     * Evict least recently used model if VRAM pressure is high
     * Returns true if a model was evicted
     */
    public boolean evictLRU() {
        if (loadedModels.isEmpty()) {
            return false;
        }

        // Find LRU model
        ModelInstance lru = loadedModels.values().stream()
                .min(Comparator.comparingLong(inst -> inst.lastUsedMs))
                .orElse(null);

        String path = lru.model.getPath();
        unloadModel(path);
        return true;

    }

    /**
     * Print service status
     */
    public void printStatus() {
        System.out.println("=".repeat(60));
        System.out.println("LLM Service Status");
        System.out.println("=".repeat(60));
        System.out.println("Loaded Models:    " + loadedModels.size());
        System.out.println("-".repeat(60));

        for (ModelInfo info : getLoadedModels()) {
            System.out.println(info);
        }

        System.out.println("=".repeat(60));
    }

    @Override
    public void close() {
        // Unload all models
        new ArrayList<>(loadedModels.keySet()).forEach(this::unloadModel);
        
        // Free backend
        LlamaBindings.free();
    }

    /**
     * Model information record
     */
    public record ModelInfo(
            String path,
            long paramCount,
            double sizeGB,
            ModelConfig modelConfig,
            long lastUsedMs
    ) {
        @Override
        public String toString() {
            long ageMs = System.currentTimeMillis() - lastUsedMs;
            return String.format("Model[%.1fB params, %.2f GB, age=%ds]",
                    paramCount / 1_000_000_000.0, sizeGB, ageMs / 1000);
        }
    }
}
