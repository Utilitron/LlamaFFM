package ffm.llama.context.state;

import ffm.llama.IntegrationTestBase;
import ffm.llama.batch.BatchDecoder;
import ffm.llama.cache.KvCacheManager;
import ffm.llama.config.ModelConfig;
import ffm.llama.batch.LlamaBatch;
import ffm.llama.context.LlamaContext;
import ffm.llama.model.LlamaModel;
import ffm.llama.service.LlmService;
import org.junit.jupiter.api.*;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class ContextStateManagerIT extends IntegrationTestBase {
    
    private static final String MODEL_RESOURCE_ENV = "LLAMA_TEST_MODEL_RESOURCE";
    
    private LlamaModel model;
    private LlamaContext context;
    private ModelConfig config;
    
    // New helper references
    private KvCacheManager kvCache;
    private BatchDecoder decoder;
    
    @BeforeEach
    void setUp() {
        String resourceName = System.getenv(MODEL_RESOURCE_ENV);
        assumeTrue(resourceName != null && !resourceName.isBlank(),
                () -> "Skipping test: environment variable " + MODEL_RESOURCE_ENV + " is not set");
        
        Path modelPath = resourceToPath(resourceName);
        config = ModelConfig.Builder.create()
                .contextSize(128)
                .batchSize(32)
                .embeddings(false)
                .build();
        model = new LlamaModel(modelPath.toString(), config);
        context = new LlamaContext(model, config);
        
        // Create helpers from the context's native pointer
        MemorySegment ctxPtr = context.ptr();
        kvCache = new KvCacheManager(ctxPtr, model);
        decoder  = new BatchDecoder(ctxPtr);
    }
    
    @AfterEach
    void tearDown() {
        if (context != null) context.close();
        if (model != null) model.close();
    }
    
    @Test
    @DisplayName("Should snapshot and restore context successfully")
    void shouldSnapshotAndRestoreContext() {
        // Prefill a few tokens using BatchDecoder
        int[] tokens = {1, 2, 3, 4};
        LlamaBatch batch = LlamaBatch.create(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            batch.add(tokens[i], i, 0, i == tokens.length - 1);
        }
        assertEquals(0, decoder.decode(batch));
        
        int posBefore = kvCache.getMaxSequencePosition(0) + 1;
        
        // Snapshot – ContextStateManager still uses LlamaContext directly (no change needed)
        var state = ContextStateManager.snapshotContext(
                new LlmService.ModelInstance(model, context, config)
        ).orElseThrow();
        assertNotNull(state);
        assertEquals(posBefore, state.getNTokens());
        
        // Clear context using KvCacheManager
        kvCache.clearKvCache();
        assertEquals(0, kvCache.getMaxSequencePosition(0) + 1);
        
        // Restore
        boolean restored = ContextStateManager.restoreContext(context, state);
        assertTrue(restored);
        assertEquals(posBefore, kvCache.getMaxSequencePosition(0) + 1);
    }
}
