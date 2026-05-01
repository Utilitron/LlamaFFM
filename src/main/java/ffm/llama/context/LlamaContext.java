package ffm.llama.context;

import ffm.llama.batch.BatchDecoder;
import ffm.llama.binding.LlamaBindings;
import ffm.llama.cache.KvCacheManager;
import ffm.llama.config.ModelConfig;
import ffm.llama.batch.LlamaBatch;
import ffm.llama.context.state.ContextStateIO;
import ffm.llama.model.LlamaModel;
import ffm.llama.utils.PerformanceMonitor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Represents a llama.cpp inference context with model configuration.
 * Manages KV cache, batch processing, and performance monitoring.
 */
public class LlamaContext implements AutoCloseable {
    
    /**
     * Maximum physical batch size (ubatch) for processing.
     * This is a hardware/performance limit enforced by llama.cpp to prevent
     * memory issues and ensure efficient batching. The physical batch size
     * is capped at 512 regardless of the logical batch size configuration.
     */
    private static final int MAX_PHYSICAL_BATCH_SIZE = 512;
    
    private final MemorySegment ctx;
    private final LlamaModel model;
    private final ModelConfig modelConfig;
    private final Arena contextArena;
    private final KvCacheManager kvCache;
    private final BatchDecoder decoder;
    private final ContextStateIO contextStateIO;
    private final PerformanceMonitor performanceMonitor;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Create a new context with default model configuration
     */
    public LlamaContext(LlamaModel model) {
        this(model, null);
    }
    
    /**
     * Create a new context with explicit model configuration
     *
     * @param model       The loaded model
     * @param modelConfig Model configuration (null = use model's config)
     */
    public LlamaContext(LlamaModel model, ModelConfig modelConfig) {
        this.model = model;
        this.contextArena = Arena.ofConfined();
        
       
        // Determine model config to use
        if (modelConfig != null) {
            this.modelConfig = modelConfig;
        } else if (model.getModelConfig() != null) {
            this.modelConfig = model.getModelConfig();
        } else {
            throw new IllegalStateException("Model configuration could not be determined");
        }
        
        try {
            // Create context parameters struct
            MemorySegment contextParams = contextArena.allocate(LlamaBindings.CONTEXT_PARAMS_LAYOUT);
            
            // Get default parameters
            MemorySegment defaultParams = (MemorySegment) LlamaBindings.llama_context_default_params.invoke(contextArena);
            
            // Copy defaults to our arena
            MemorySegment.copy(defaultParams, 0, contextParams, 0, LlamaBindings.CONTEXT_PARAMS_LAYOUT.byteSize());
            
            // Apply model configuration
            applyModelConfigToContextParams(contextParams);
            
            // Create context
            this.ctx = (MemorySegment) LlamaBindings.llama_init_from_model.invoke(model.ptr(), contextParams);
            
            if (ctx == MemorySegment.NULL) {
                try { contextArena.close(); } catch (Exception ignored) {}
                throw new RuntimeException("Failed to create context");
            }
            
            this.kvCache = new KvCacheManager(ctx, model);
            this.decoder = new BatchDecoder(ctx);
            this.contextStateIO = new ContextStateIO(ctx);
            this.performanceMonitor = new PerformanceMonitor(ctx);
            
        } catch (Throwable t) {
            try { contextArena.close(); } catch (Exception ignored) {}
            throw new RuntimeException("Failed to create context", t);
        }
    }
    
    /**
     * Apply model configuration to context parameters struct
     */
    private void applyModelConfigToContextParams(MemorySegment contextParams) {
        try {
            // Context size
            LlamaBindings.CONTEXT_N_CTX.set(contextParams, 0L, modelConfig.getContextSize());
            
            // Batch size
            LlamaBindings.CONTEXT_N_BATCH.set(contextParams, 0L, modelConfig.getBatchSize());
            
            // Physical batch size
            LlamaBindings.CONTEXT_N_UBATCH.set(contextParams, 0L, Math.min(MAX_PHYSICAL_BATCH_SIZE, modelConfig.getBatchSize()));
            
            // CPU threads
            LlamaBindings.CONTEXT_N_THREADS.set(contextParams, 0L, modelConfig.getCpuThreads());
            
            // Batch threads
            LlamaBindings.CONTEXT_N_THREADS_BATCH.set(contextParams, 0L, modelConfig.getCpuThreads());
            
            // KV cache offloading
            LlamaBindings.CONTEXT_OFFLOAD_KQV.set(contextParams, 0L, (byte) (modelConfig.isOffloadKvToGpu() ? 1 : 0));
            
            // Flash attention (INT, correct field name)
            LlamaBindings.CONTEXT_FLASH_ATTN_TYPE.set(contextParams, 0L, modelConfig.isFlashAttention() ? 1 : 0);
            
            // Defragmentation threshold
            LlamaBindings.CONTEXT_DEFRAG_THOLD.set(contextParams, 0L, modelConfig.getDefragThreshold());
            
            // Performance metrics
            LlamaBindings.CONTEXT_NO_PERF.set(contextParams, 0L, (byte) 0);
            
            // Embeddings
            LlamaBindings.CONTEXT_EMBEDDINGS.set(contextParams, 0L, (byte) (modelConfig.isEmbeddings() ? 1 : 0));
            
            // KVCache Types
            LlamaBindings.CONTEXT_K_CACHE_TYPE.set(contextParams, 0L, (byte) (modelConfig.getCacheTypeK().getNativeId()));
            LlamaBindings.CONTEXT_V_CACHE_TYPE.set(contextParams, 0L, (byte) (modelConfig.getCacheTypeV().getNativeId()));
            
            
        } catch (Throwable t) {
            throw new RuntimeException("Failed to apply model config to context params", t);
        }
    }
    
    /**
     * Get the native pointer to the context
     */
    public MemorySegment ptr() { ensureNotClosed(); return ctx; }
    
    /**
     * Get the associated model
     */
    public LlamaModel getModel() {
        return model;
    }
    
    /**
     * Get model configuration
     */
    public ModelConfig getModelConfig() {
        return modelConfig;
    }
    
    /**
     * Get KV cache manager
     */
    public KvCacheManager kvCache() { ensureNotClosed(); return kvCache; }
    
    /**
     * Get batch decoder
     */
   
    public BatchDecoder decoder() { ensureNotClosed(); return decoder; }
    
    /**
     * Get context state IO
     */
    public ContextStateIO contextStateIO() { ensureNotClosed(); return contextStateIO; }
    
    /**
     * Get performance monitor
     */
    public PerformanceMonitor performanceMonitor() { ensureNotClosed(); return performanceMonitor; }
    
    // ============================================================================
    // UTILITIES
    // ============================================================================
    
    /**
     * Print context information to console
     */
    public void printInfo() {
        ensureNotClosed();
        System.out.println("=".repeat(60));
        System.out.println("Context Information");
        System.out.println("=".repeat(60));
        System.out.println("Context Size:     " + modelConfig.getContextSize());
        System.out.println("Batch Size:       " + modelConfig.getBatchSize());
        System.out.println("CPU Threads:      " + modelConfig.getCpuThreads());
        System.out.println("GPU Layers:       " + modelConfig.getGpuLayers());
        System.out.println("KV on GPU:        " + modelConfig.isOffloadKvToGpu());
        System.out.println("Flash Attn:       " + modelConfig.isFlashAttention());
        kvCache.printInfo();
        System.out.println("=".repeat(60));
        System.out.println("\n\n");
    }
    
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        
        try {
            // Free the context
            LlamaBindings.llama_free.invoke(ctx);
        } catch (Throwable t) {
            // Log but don't throw - we're in cleanup
            System.err.println("Warning: Failed to free context: " + t.getMessage());
        } finally {
            // Always close the arena
            contextArena.close();
        }
    }
    
    /**
     * Checks if this context has been closed.
     *
     * @return true if close() has been called
     */
    public boolean isClosed() {
        return closed.get();
    }
    
    /**
     * Ensures this context is not closed.
     *
     * @throws IllegalStateException if context is closed
     */
    private void ensureNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("Context has been closed");
        }
    }
}