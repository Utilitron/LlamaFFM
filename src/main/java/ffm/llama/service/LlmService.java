package ffm.llama.service;

import ffm.llama.binding.LlamaBindings;
import ffm.llama.binding.LlamaPoolingType;
import ffm.llama.config.ModelConfig;
import ffm.llama.model.LlamaBatch;
import ffm.llama.model.LlamaContext;
import ffm.llama.model.LlamaModel;
import ffm.llama.sampling.LlamaSampler;
import ffm.llama.utils.TemplateDetector;
import ffm.llama.utils.LlmToolGrammar;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

/**
 * High-level LLM service
 */
public class LlmService implements AutoCloseable {

    //Interface for all messages passed to the LlmService
    public interface LlmMessage {
        String role();
        String content();
    }

    //Interface for all messages passed to the LlmService with tools
    public interface LlmMessageWithTools extends LlmMessage {
        String toolDefinitions();
    }

    //Interface for all messages passed to the LlmService with tools
    public interface LlmToolMessage extends LlmMessage {
        String toolCallId();
        String toolName();
    }

    // Simple container for the chat message
    public record ChatMessage(
            String role,
            String content
    ) implements LlmMessage {}

    // Simple container for the chat message
    public record ChatMessageWithTools(
            String role,
            String content,
            String toolDefinitions
    ) implements LlmMessageWithTools {}

    public record ToolMessage(
            String role,
            String toolCallId,
            String toolName,
            String content
    ) implements LlmToolMessage {}

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
        String modelName = Paths.get(modelPath).getFileName().toString();
        // Check if already loaded
        if (loadedModels.containsKey(modelName)) {
            return modelName;
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
            loadedModels.put(modelName, instance);

            System.out.println("Loaded model: " + modelName);
            model.printInfo();
            context.printInfo();

            return modelName;

        } catch (Exception e) {
            throw new RuntimeException("Failed to load model: " + modelName, e);
        }
    }

    /**
     * Unload a model to free VRAM
     */
    public void unloadModel(String modelPath) {
        String modelName = Paths.get(modelPath).getFileName().toString();
        ModelInstance instance = loadedModels.remove(modelName);
        if (instance != null) {
            instance.close();
            System.out.println("Unloaded model: " + modelName);
        }
    }

    /**
     * Apply a chat template (Jinja2) to a list of messages.
     * If a template string is provided, it is used; otherwise the model's default template is used.
     * If any message is an instance of LlmMessageWithTools, its tool definitions are injected
     * into the system message (or a new system message is created) before applying the template.
     *
     * @param template The Jinja2 template string (may be null to use the model's default)
     * @param history The list of messages
     * @param addAssistant Whether to add an assistant generation prompt at the end
     * @return The formatted prompt string
     */
    public String applyChatTemplate(String template, List<? extends LlmMessage> history, boolean addAssistant) {
        try (Arena arena = Arena.ofConfined()) {
            String toolDefinitions = null;
            List<LlmMessage> bakedHistory = new ArrayList<>();

            // Extract tool definitions from any LlmMessageWithTools and collect plain messages
            for (LlmMessage msg : history) {
                if (msg instanceof LlmMessageWithTools toolMsg) {
                    if (toolDefinitions == null) {
                        toolDefinitions = LlmToolGrammar.injectTools(TemplateDetector.detectTemplate(template), toolMsg.toolDefinitions());
                    }
                    bakedHistory.add(new ChatMessage(msg.role(), msg.content()));
                } else if (msg instanceof ToolMessage toolMsg) {
                    bakedHistory.add(toolMsg);
                } else {
                    bakedHistory.add(msg);
                }
            }

            // Inject tool definitions into the system message (or create one)
            if (toolDefinitions != null && !toolDefinitions.isEmpty()) {
                boolean injected = false;

                for (int i = 0; i < bakedHistory.size(); i++) {
                    LlmMessage msg = bakedHistory.get(i);

                    if ("system".equals(msg.role())) {
                        bakedHistory.set(i, new ChatMessage("system", msg.content() + " " + toolDefinitions));
                        injected = true;
                        break;
                    }
                }

                if (!injected) bakedHistory.add(0, new ChatMessage("system", toolDefinitions));
            }

            int nativeCount = bakedHistory.size();
            MemorySegment chatArray = arena.allocate(LlamaBindings.CHAT_LAYOUT, nativeCount);

            // Allocate it as a C-string. If null, we fall back to NULL for the internal default.
            MemorySegment templateSeg = (template != null) ? arena.allocateFrom(template) : MemorySegment.NULL;

            for (int i = 0; i < nativeCount; i++) {
                LlmMessage msg = bakedHistory.get(i);
                MemorySegment currentStruct = chatArray.asSlice(i * LlamaBindings.CHAT_LAYOUT.byteSize());

                currentStruct.set(ValueLayout.ADDRESS, 0, arena.allocateFrom(msg.role()));
                currentStruct.set(ValueLayout.ADDRESS, 8, arena.allocateFrom(msg.content()));
            }

            // First Pass: Get the required buffer size
            int requiredSize = (int) LlamaBindings.llama_chat_apply_template.invokeExact(
                    templateSeg, chatArray, (long) nativeCount, addAssistant, MemorySegment.NULL, 0
            );
            if (requiredSize < 0) throw new RuntimeException("Template application failed with error code: " + requiredSize);

            // Second Pass: Allocate the buffer and fill it
            MemorySegment buffer = arena.allocate(ValueLayout.JAVA_BYTE, requiredSize);

            int actualSize = (int) LlamaBindings.llama_chat_apply_template.invokeExact(
                    templateSeg, chatArray, (long) nativeCount, addAssistant, buffer, requiredSize
            );

            if (actualSize <= 0) throw new RuntimeException("Template application failed during formatting.");

            byte[] bytes = buffer.asSlice(0, actualSize).toArray(ValueLayout.JAVA_BYTE);
            int effectiveLength = (bytes.length > 0 && bytes[bytes.length - 1] == 0) ? bytes.length - 1 : bytes.length;

            return new String(bytes, 0, effectiveLength, StandardCharsets.UTF_8);

        } catch (Throwable t) {
            throw new RuntimeException("Failed to apply chat template", t);
        }
    }

    /**
     * Generate text with greedy sampling
     * 
     * @param modelName Model identifier
     * @param conversation The structured chat history (System, User, Assistant roles)
     * @return Generated text
     */
    public String generate(String modelName, List<? extends LlmMessage> conversation) {
        return generate(modelName, conversation, LlamaSampler.SamplerConfig.greedy());
    }

    /**
     * Generate text with custom sampling configuration
     * 
     * @param modelName Model identifier
     * @param conversation The structured chat history (System, User, Assistant roles)
     * @param samplerConfig Sampling strategy
     * @return Generated text
     */
    public String generate(String modelName, List<? extends LlmMessage> conversation, LlamaSampler.SamplerConfig samplerConfig) {
        StringBuilder result = new StringBuilder();
        generateStreaming(modelName, conversation, samplerConfig, result::append);
        return result.toString();
    }

    /**
     * Generate text with streaming callback
     * Calls the callback for each generated token
     * 
     * @param modelName Model identifier
     * @param conversation The structured chat history (System, User, Assistant roles)
     * @param samplerConfig Sampling strategy
     * @param callback Function called with each generated token
     */
    public void generateStreaming(
            String modelName,
            List<? extends LlmMessage> conversation,
            LlamaSampler.SamplerConfig samplerConfig,
            Consumer<String> callback
    ) {
        ModelInstance instance = loadedModels.get(modelName);
        if (instance == null) {
            throw new IllegalArgumentException("Model not loaded: " + modelName);
        }

        instance.updateLastUsed();
        LlamaModel model = instance.model;
        LlamaContext ctx = instance.context;

        String formattedPrompt = applyChatTemplate(model.getChatTemplate(), conversation, true);

        try (LlamaSampler sampler = new LlamaSampler(samplerConfig, model.vocabPtr())) {

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
            throw new RuntimeException("Generation failed for model: " + modelName, e);
        }
    }

    /**
     * Generate embeddings for text
     * Requires model loaded with embeddings=true in context params
     * Automatically detects pooling type and uses appropriate strategy
     * 
     * @param modelName Model identifier (should be embedding model like nomic-embed)
     * @param text Text to embed
     * @return Embedding vector as float array
     */
    public float[] embed(String modelName, String text) {
        ModelInstance instance = loadedModels.get(modelName);
        if (instance == null) {
            throw new IllegalArgumentException("Model not loaded: " + modelName);
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
    public ModelInfo getModelInfo(String modelName) {
        ModelInstance instance = loadedModels.get(modelName);
        if (instance == null) {
            return null;
        }

        LlamaModel model = instance.model;
        return new ModelInfo(
                modelName,
                TemplateDetector.getTemplateName(model.getChatTemplate()),
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
                .map(inst -> getModelInfo(Paths.get(inst.model.getPath()).getFileName().toString()))
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
            String fileName,
            String templateName,
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
