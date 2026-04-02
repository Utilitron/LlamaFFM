package ffm.llama.service;

import ffm.llama.binding.LlamaBindings;
import ffm.llama.binding.LlamaPoolingType;
import ffm.llama.config.ModelConfig;
import ffm.llama.model.LlamaBatch;
import ffm.llama.model.LlamaContext;
import ffm.llama.model.LlamaModel;
import ffm.llama.sampling.LlamaSampler;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

/**
 * High-level LLM service
 */
public class LlmService implements AutoCloseable {

    // Simple container for the chat message
    public record ChatMessage(String role, String content) {}

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

    public String applyChatTemplate(List<ChatMessage> history, boolean addAssistant) {
        try (Arena arena = Arena.ofConfined()) {
            int messageCount = history.size();

            // Allocate an array of llama_chat_message structs (16 bytes each)
            MemorySegment chatArray = arena.allocate(LlamaBindings.CHAT_LAYOUT, messageCount);

            for (int i = 0; i < messageCount; i++) {
                ChatMessage msg = history.get(i);

                // Calculate the offset for the i-th struct in the array
                MemorySegment currentStruct = chatArray.asSlice(i * LlamaBindings.CHAT_LAYOUT.byteSize());

                // Allocate C-strings (null-terminated) for role and content
                MemorySegment roleSeg = arena.allocateFrom(msg.role());
                MemorySegment contentSeg = arena.allocateFrom(msg.content());

                // Write the pointers into the struct
                currentStruct.set(ValueLayout.ADDRESS, 0, roleSeg);  // Offset 0: role
                currentStruct.set(ValueLayout.ADDRESS, 8, contentSeg); // Offset 8: content
            }

            // First Pass: Get the required buffer size
            // Passing NULL (MemorySegment.NULL) for the template string uses the GGUF's internal default.
            int requiredSize = (int) LlamaBindings.llama_chat_apply_template.invokeExact(
                    MemorySegment.NULL,
                    chatArray,
                    (long) messageCount,
                    addAssistant,
                    MemorySegment.NULL,
                    0
            );

            if (requiredSize < 0) {
                throw new RuntimeException("Template application failed with error code: " + requiredSize);
            }

            // Second Pass: Allocate the buffer and fill it
            // llama.cpp returns the size including the null terminator
            MemorySegment buffer = arena.allocate(ValueLayout.JAVA_BYTE, requiredSize);

            int actualSize = (int) LlamaBindings.llama_chat_apply_template.invokeExact(
                    MemorySegment.NULL,
                    chatArray,
                    (long) messageCount,
                    addAssistant,
                    buffer,
                    requiredSize
            );

            if (actualSize <= 0) {
                throw new RuntimeException("Template application failed during formatting.");
            }

            // Use the returned size to grab exactly the bytes we need.
            // We subtract 1 if the returned size includes the null terminator (standard for llama.cpp)
            byte[] bytes = buffer.asSlice(0, actualSize).toArray(ValueLayout.JAVA_BYTE);

            // If the last byte is a null terminator, strip it before making the Java String
            int effectiveLength = (bytes.length > 0 && bytes[bytes.length - 1] == 0)
                    ? bytes.length - 1
                    : bytes.length;

            // Convert back to a Java String (FFM handles the UTF-8 conversion)
            return new String(bytes, 0, effectiveLength, StandardCharsets.UTF_8);

        } catch (Throwable t) {
            throw new RuntimeException("Failed to apply chat template", t);
        }
    }

    /**
     * Generate text with greedy sampling
     * 
     * @param modelPath Model identifier
     * @param conversation The structured chat history (System, User, Assistant roles)
     * @return Generated text
     */
    public String generate(String modelPath, List<ChatMessage> conversation) {
        return generate(modelPath, conversation, LlamaSampler.SamplerConfig.greedy());
    }

    /**
     * Generate text with custom sampling configuration
     * 
     * @param modelPath Model identifier
     * @param conversation The structured chat history (System, User, Assistant roles)
     * @param samplerConfig Sampling strategy
     * @return Generated text
     */
    public String generate(String modelPath, List<ChatMessage> conversation, LlamaSampler.SamplerConfig samplerConfig) {
        StringBuilder result = new StringBuilder();
        generateStreaming(modelPath, conversation, samplerConfig, result::append);
        return result.toString();
    }

    /**
     * Generate text with streaming callback
     * Calls the callback for each generated token
     * 
     * @param modelPath Model identifier
     * @param conversation The structured chat history (System, User, Assistant roles)
     * @param samplerConfig Sampling strategy
     * @param callback Function called with each generated token
     */
    public void generateStreaming(
            String modelPath,
            List<ChatMessage> conversation,
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

        String formattedPrompt = applyChatTemplate(conversation, true);

        try (LlamaSampler sampler = new LlamaSampler(samplerConfig)) {

            // Clear KV cache for fresh generation
            ctx.clearKvCache();

            // Tokenize prompt
            int[] promptTokens = model.tokenize(formattedPrompt, true, true);

            // Calculate how many slots are left in your context window
            int maxPossibleTokens = Math.max(1, ctx.getModelConfig().getContextSize() - promptTokens.length);

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

            while (nDecoded < maxPossibleTokens) {
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

            if (nDecoded >= maxPossibleTokens) throw new IllegalStateException("Ran out of tokens");

        } catch (Exception e) {
            throw new RuntimeException("Generation failed for model: " + modelPath, e);
        }
    }

    /**
     * Generate embeddings for text
     * Requires model loaded with embeddings=true in context params
     * Automatically detects pooling type and uses appropriate strategy
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

        // Validate that model is configured for embeddings
        if (!instance.modelConfig.isEmbeddings()) {
            throw new IllegalStateException("Model not configured for embeddings.");
        }

        instance.updateLastUsed();
        LlamaModel model = instance.model;
        LlamaContext ctx = instance.context;

        try {
            // Get the embedding size
            int n_embd = (int) LlamaBindings.llama_model_n_embd.invokeExact(model.ptr());

            // Get pooling type to determine strategy
            int poolingType = (int) LlamaBindings.llama_pooling_type.invokeExact(ctx.ptr());
            boolean isNone = poolingType == LlamaPoolingType.NONE.getValue();

            int[] tokens = model.tokenize(text, true, false);

            // Clear KV cache for fresh generation
            ctx.clearKvCache();

            // Process batch - enable logits for last token
            try (LlamaBatch batch = LlamaBatch.forTokens(tokens, 0, 0, !isNone)) {
                int ret = ctx.decode(batch);
                if (ret != 0) {
                    throw new RuntimeException("Failed to decode batch (error code: " + ret + ")");
                }
            }

            // Retrieve computed embeddings
            if (poolingType == LlamaPoolingType.NONE.getValue()) {
                // Access the full token-level embedding buffer
                MemorySegment allEmbeds = (MemorySegment) LlamaBindings.llama_get_embeddings.invokeExact(ctx.ptr());
                // Offset to the last token: (tokens.length - 1) * n_embd * sizeof(float)
                long offset = (long) (tokens.length - 1) * n_embd * Float.BYTES;
                return copyEmbedding(allEmbeds.asSlice(offset), n_embd);
            } else {
                // Access the pooled sequence-level embedding buffer
                MemorySegment seqEmbed = (MemorySegment) LlamaBindings.llama_get_embeddings_seq.invokeExact(ctx.ptr(), 0);
                if (seqEmbed.address() == 0L) {
                    throw new RuntimeException("Model pooling failed to produce a sequence embedding.");
                }
                return copyEmbedding(seqEmbed, n_embd);
            }

        } catch (Throwable e) {
            throw new RuntimeException("Embedding generation failed", e);
        }
    }

    /**
     * Copy embedding from memory segment to float array
     */
    private float[] copyEmbedding(MemorySegment embSeg, int n_embd) {
        MemorySegment safe = embSeg.reinterpret((long) n_embd * Float.BYTES);
        float[] result = new float[n_embd];
        for (int i = 0; i < n_embd; i++) {
            result[i] = safe.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return result;
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
