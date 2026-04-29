package ffm.llama.service;

import ffm.llama.binding.LlamaBindings;
import ffm.llama.config.ModelConfig;
import ffm.llama.enums.PoolingType;
import ffm.llama.message.*;
import ffm.llama.model.LlamaBatch;
import ffm.llama.model.LlamaContext;
import ffm.llama.model.LlamaModel;
import ffm.llama.model.state.CachedContextState;
import ffm.llama.model.state.ContextStateManager;
import ffm.llama.sampling.LlamaSampler;
import ffm.llama.session.GenerationSession;
import ffm.llama.session.strategy.ContextStrategy;
import ffm.llama.utils.LlmToolGrammar;
import ffm.llama.utils.NativeMemoryUtils;
import ffm.llama.utils.TemplateDetector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

/**
 * High-level LLM service
 */
public class LlmService implements AutoCloseable {
    
    // Initialize llama.cpp backend once
    static {
        LlamaBindings.init();
    }
    
    // Model registry - maps model names to loaded instances
    private final Map<String, ModelInstance> loadedModels = new ConcurrentHashMap<>();
    // Evicted state cache - stores KV cache snapshots in RAM
    private final Map<String, CachedContextState> evictedStates = new ConcurrentHashMap<>();
    private boolean verbose = false;
    
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
     * Attempts to restore KV cache from evicted state if available and compatible.
     *
     * @param modelPath   Path to .gguf model file
     * @param modelConfig Model configuration (null = default settings)
     * @return Model identifier for subsequent calls
     */
    public String loadModel(String modelPath, ModelConfig modelConfig) {
        String modelName = Paths.get(modelPath).getFileName().toString();
        
        // Check if already loaded
        if (loadedModels.containsKey(modelName)) {
            if (verbose) {
                System.out.println("Model already loaded: " + modelName);
            }
            return modelName;
        }
        
        try {
            // Load model
            LlamaModel model = new LlamaModel(modelPath, modelConfig);
            
            // Determine final config (use default if not provided)
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
            
            // Check for cached state and restore if compatible
            CachedContextState cachedState = evictedStates.get(modelPath);
            boolean restored = false;
            
            if (cachedState != null) {
                if (cachedState.isCompatibleWith(modelPath, finalConfig)) {
                    // Attempt to restore cached KV cache
                    restored = ContextStateManager.restoreContext(context, cachedState);
                    
                    if (restored) {
                        System.out.printf("Restored KV cache for %s (%.2f MB, saved %d ms ago)%n", modelName, cachedState.getSizeMB(), cachedState.getAgeMs());
                        
                        // Remove from cache after successful restore
                        evictedStates.remove(modelPath);
                    } else {
                        System.err.println("Failed to restore cached state for " + modelName);
                        // Discard incompatible/corrupted state
                        evictedStates.remove(modelPath);
                    }
                } else {
                    System.out.println("Cached state for " + modelName + " is incompatible with current config, discarding");
                    evictedStates.remove(modelPath);
                }
            }
            
            // Register instance
            ModelInstance instance = new ModelInstance(model, context, finalConfig);
            instance.restoredFromCache = restored;
            loadedModels.put(modelName, instance);
            
            System.out.println("Loaded model: " + modelName);
            if (verbose) {
                System.out.println("Loaded model: " + modelName + (restored ? " [KV cache restored]" : ""));
                model.printInfo();
                context.printInfo();
            }
            
            return modelName;
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to load model: " + modelName, e);
        }
    }
    
    /**
     * Unload a model and free its resources.
     * Does NOT snapshot state (use evictLRU for that).
     *
     * @param modelPath Path to the model to unload
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
     * @param template     The Jinja2 template string (may be null to use the model's default)
     * @param history      The list of messages
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
                    
                    if (msg.role() == MessageRole.SYSTEM) {
                        bakedHistory.set(i, new ChatMessage(MessageRole.SYSTEM, msg.content() + " " + toolDefinitions));
                        injected = true;
                        break;
                    }
                }
                
                if (!injected) bakedHistory.addFirst(new ChatMessage(MessageRole.SYSTEM, toolDefinitions));
            }
            
            int nativeCount = bakedHistory.size();
            MemorySegment chatArray = arena.allocate(LlamaBindings.CHAT_LAYOUT, nativeCount);
            
            // Allocate it as a C-string. If null, we fall back to NULL for the internal default.
            MemorySegment templateSeg = (template != null) ? arena.allocateFrom(template) : MemorySegment.NULL;
            
            for (int i = 0; i < nativeCount; i++) {
                LlmMessage msg = bakedHistory.get(i);
                MemorySegment currentStruct = chatArray.asSlice(i * LlamaBindings.CHAT_LAYOUT.byteSize());
                
                currentStruct.set(ValueLayout.ADDRESS, 0, arena.allocateFrom(msg.role().getValue()));
                currentStruct.set(ValueLayout.ADDRESS, 8, arena.allocateFrom(msg.content()));
            }
            
            // First Pass: Get the required buffer size
            int requiredSize = (int) LlamaBindings.llama_chat_apply_template.invokeExact(templateSeg, chatArray, (long) nativeCount, addAssistant, MemorySegment.NULL, 0);
            if (requiredSize < 0)
                throw new RuntimeException("Template application failed with error code: " + requiredSize);
            
            // Second Pass: Allocate the buffer and fill it
            MemorySegment buffer = arena.allocate(ValueLayout.JAVA_BYTE, requiredSize);
            
            int actualSize = (int) LlamaBindings.llama_chat_apply_template.invokeExact(templateSeg, chatArray, (long) nativeCount, addAssistant, buffer, requiredSize);
            if (actualSize <= 0) throw new RuntimeException("Template application failed during formatting.");
            
            return NativeMemoryUtils.readCStringExact(buffer, actualSize);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to apply chat template", t);
        }
    }
    
    /**
     * Generate text with greedy sampling
     *
     * @param modelName    Model identifier
     * @param conversation The structured chat history (System, User, Assistant roles)
     * @return Generated text
     */
    public String generate(String modelName, List<? extends LlmMessage> conversation) {
        return generate(modelName, conversation, LlamaSampler.SamplerConfig.greedy());
    }
    
    /**
     * Generate text with custom sampling configuration
     *
     * @param modelName     Model identifier
     * @param conversation  The structured chat history (System, User, Assistant roles)
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
     * @param modelName     Model identifier
     * @param conversation  The structured chat history (System, User, Assistant roles)
     * @param samplerConfig Sampling strategy
     * @param callback      Function called with each generated token
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
        
        // Tokenize prompt
        int[] promptTokens = model.tokenize(formattedPrompt, true, true);
        
        // Get context size from model config
        int contextSize = ctx.getModelConfig().getContextSize();
        
        // Context overflow - prevents native crashes and undefined behavior
        if (promptTokens.length > contextSize) {
            throw new IllegalStateException(String.format(
                    "Prompt exceeds context window: %d > %d", promptTokens.length, contextSize));
        }
        
        // Use GenerationSession for the prefill and generation loop
        try (GenerationSession session = GenerationSession.builder()
                .model(model)
                .context(ctx)
                .sampler(samplerConfig)
                .contextStrategy(ContextStrategy.slidingWindow(contextSize, 0.5, 256))
                .verbose(verbose)
                .build()) {
            
            session.reset();            // clear KV cache for fresh generation
            session.prefill(promptTokens);
            session.generate(callback, 0);   // 0 = unlimited, until EOS
            
            if (verbose) System.out.println(session.getMetrics().summary());
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
     * @param text      Text to embed
     * @return Embedding vector as float array
     */
    public float[] embed(String modelName, String text) {
        return embed(modelName, text, false);
    }
    
    /**
     * Generate embeddings for text
     * Requires model loaded with embeddings=true in context params
     * Automatically detects pooling type and uses appropriate strategy
     *
     * @param modelName Model identifier (should be embedding model like nomic-embed)
     * @param text      Text to embed
     * @param truncate  If true, silently truncates to fit context; otherwise throws on overflow
     * @return Embedding vector as float array
     */
    public float[] embed(String modelName, String text, boolean truncate) {
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
            boolean isNone = poolingType == PoolingType.NONE.getValue();
            
            // Hard limit = batch size
            int maxTokens = ctx.getModelConfig().getBatchSize();
            int[] tokens = model.tokenize(text, true, false);
            
            // Enforce limit
            if (tokens.length > maxTokens) {
                if (truncate) {
                    // Truncate to maxTokens – keep the first maxTokens tokens
                    tokens = Arrays.copyOf(tokens, maxTokens);
                    if (verbose) {
                        System.out.printf("[LlmService] Embedding truncated from %d to %d tokens%n", tokens.length, maxTokens);
                    }
                } else {
                    throw new IllegalStateException(String.format("Text length %d tokens exceeds maximum %d. Use truncate=true or split the input.", tokens.length, maxTokens));
                }
            }
            
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
            if (poolingType == PoolingType.NONE.getValue()) {
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
                model.getLayerCount(),
                model.getEmbeddingSize(),
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
     * Evict a specific model, optionally saving its KV cache state.
     *
     * @param modelName The model identifier (filename)
     * @return true if evicted, false if not loaded
     */
    public boolean evictModel(String modelName) {
        ModelInstance instance = loadedModels.get(modelName);
        if (instance == null) return false;
        
        String modelPath = instance.model.getPath();
        
        ContextStateManager.snapshotContext(instance).ifPresentOrElse(state -> {
                    evictedStates.put(modelPath, state);
                    if (verbose)
                        System.out.printf("[LlmService] KV snapshot saved for %s (%.2f MB)%n", modelName, state.getSizeMB());
                }, () -> System.err.println("[LlmService] Failed to snapshot " + modelName)
        );
        
        loadedModels.remove(modelName);
        instance.close();
        
        if (verbose) System.out.println("Evicted model: " + modelName);
        return true;
    }
    
    /**
     * Evict least recently used model with KV cache snapshot.
     *
     * @return true if a model was evicted, false if no models loaded
     */
    public boolean evictLRU() {
        if (loadedModels.isEmpty()) return false;
        
        // Find LRU model
        Map.Entry<String, ModelInstance> lruEntry = loadedModels.entrySet().stream()
                .min(Comparator.comparingLong(e -> e.getValue().lastUsedMs))
                .orElse(null);
        
        if (lruEntry == null) return false;
        
        String modelName = lruEntry.getKey();
        
        return evictModel(modelName);
    }
    
    /**
     * Get information about cached states
     */
    public Map<String, CachedContextState> getEvictedStates() {
        return new HashMap<>(evictedStates);
    }
    
    /**
     * Manually clear all cached states to free RAM
     */
    public void clearEvictedStates() {
        int count = evictedStates.size();
        evictedStates.clear();
        
        if (verbose) {
            System.out.printf("Cleared " + count + " cached states");
        }
    }
    
    /**
     * Manually clear a specific cached state
     */
    public boolean clearEvictedState(String modelPath) {
        CachedContextState removed = evictedStates.remove(modelPath);
        return removed != null;
    }
    
    /**
     * Get total memory used by cached states
     */
    public double getTotalCachedStateSizeMB() {
        return evictedStates.values().stream()
                .mapToDouble(CachedContextState::getSizeMB)
                .sum();
    }
    
    public boolean isVerbose() {
        return verbose;
    }
    
    /**
     * Enable or disable verbose console output
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
    
    /**
     * Get all loaded models
     */
    public Set<String> getLoadedModelNames() {
        return new HashSet<>(loadedModels.keySet());
    }
    
    /**
     * Print service status including cached states
     */
    public void printStatus() {
        System.out.println("=".repeat(60));
        System.out.println("LLM Service Status (Enhanced with KV Cache Persistence)");
        System.out.println("=".repeat(60));
        System.out.println("Loaded Models:       " + loadedModels.size());
        System.out.println("Cached States:       " + evictedStates.size());
        System.out.println("Cached State Memory: " + String.format("%.2f MB", getTotalCachedStateSizeMB()));
        System.out.println("-".repeat(60));
        
        // Show loaded models
        for (Map.Entry<String, ModelInstance> entry : loadedModels.entrySet()) {
            ModelInstance inst = entry.getValue();
            System.out.printf("  [LOADED] %s %s(age: %d ms)%n",
                    entry.getKey(), inst.restoredFromCache ? "[RESTORED] " : "", System.currentTimeMillis() - inst.lastUsedMs
            );
        }
        
        // Show cached states
        for (Map.Entry<String, CachedContextState> entry : evictedStates.entrySet()) {
            CachedContextState state = entry.getValue();
            System.out.printf("  [CACHED] %s (%.2f MB, age: %d ms)%n",
                    Paths.get(entry.getKey()).getFileName(), state.getSizeMB(), state.getAgeMs()
            );
        }
        
        System.out.println("=".repeat(60));
    }
    
    @Override
    public void close() {
        // Unload all models
        new ArrayList<>(loadedModels.keySet()).forEach(name -> {
            ModelInstance inst = loadedModels.remove(name);
            if (inst != null) {
                inst.close();
            }
        });
        
        // Clear cached states
        evictedStates.clear();
        
        // Free backend
        LlamaBindings.free();
    }
    
    /**
     * Model instance wrapper
     */
    public static class ModelInstance {
        public final LlamaModel model;
        public final LlamaContext context;
        public final ModelConfig modelConfig;
        long lastUsedMs;
        
        // Track if this instance was restored from a cached state
        boolean restoredFromCache;
        
        public ModelInstance(LlamaModel model, LlamaContext context, ModelConfig modelConfig) {
            this.model = model;
            this.context = context;
            this.modelConfig = modelConfig;
            this.lastUsedMs = System.currentTimeMillis();
            this.restoredFromCache = false;
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
     * Model information record
     */
    public record ModelInfo(
            String fileName,
            String templateName,
            long paramCount,
            double sizeGB,
            int layerCount,
            int embeddingSize,
            ModelConfig modelConfig,
            long lastUsedMs
    ) {
        @Override
        public String toString() {
            long ageMs = System.currentTimeMillis() - lastUsedMs;
            return String.format("Model[%.1fB params, %.2f GB, layers=%d, embd=%d, age=%ds]",
                    paramCount / 1_000_000_000.0, sizeGB, layerCount, embeddingSize, ageMs / 1000);
        }
    }
}