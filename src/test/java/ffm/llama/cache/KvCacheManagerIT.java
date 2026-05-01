package ffm.llama.cache;

import ffm.llama.IntegrationTestBase;
import ffm.llama.batch.BatchDecoder;
import ffm.llama.batch.LlamaBatch;
import ffm.llama.config.ModelConfig;
import ffm.llama.context.LlamaContext;
import ffm.llama.model.LlamaModel;
import org.junit.jupiter.api.*;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class KvCacheManagerIT extends IntegrationTestBase {
    
    private static LlamaModel sharedModel;
    private LlamaContext context;
    private KvCacheManager kvCache;
    private BatchDecoder decoder;
    private ModelConfig config;
    private int[] promptTokens;
    
    @BeforeAll
    static void loadModel() {
        String resource = System.getenv("LLAMA_TEST_MODEL_RESOURCE");
        assumeTrue(resource != null && !resource.isBlank(),
                () -> "Skipping test: LLAMA_TEST_MODEL_RESOURCE is not set");
        Path modelPath = resourceToPath(resource);
        sharedModel = new LlamaModel(modelPath.toString());
    }
    
    @AfterAll
    static void freeModel() {
        if (sharedModel != null) {
            sharedModel.close();
            sharedModel = null;
        }
    }
    
    @BeforeEach
    void setUp() {
        config = ModelConfig.Builder.create()
                .contextSize(128)
                .batchSize(32)
                .cpuThreads(1)
                .embeddings(false)
                .build();
        context = new LlamaContext(sharedModel, config);
        
        // Create the KvCacheManager using the context's native pointer
        kvCache = new KvCacheManager(context.ptr(), sharedModel);
        decoder = new BatchDecoder(context.ptr());
        
        promptTokens = sharedModel.tokenize("Hello llama", true, false);
        assumeTrue(promptTokens.length > 0, "Model must support tokenizing 'Hello llama'");
    }
    
    @AfterEach
    void tearDown() {
        if (context != null) {
            context.close();
            context = null;
        }
    }
    
    // Helper to prefill tokens
    private void prefill() {
        LlamaBatch batch = LlamaBatch.create(promptTokens.length);
        for (int i = 0; i < promptTokens.length; i++) {
            batch.add(promptTokens[i], i, 0, i == promptTokens.length - 1);
        }
        assertEquals(0, decoder.decode(batch));
    }
    
    @Test
    @DisplayName("Should report correct max sequence position after prefill")
    void shouldReportMaxPosition() {
        prefill();
        assertEquals(promptTokens.length - 1, kvCache.getMaxSequencePosition(0));
    }
    
    @Test
    @DisplayName("Should clear KV cache and reset position")
    void shouldClearKvCache() {
        prefill();
        kvCache.clearKvCache();
        assertEquals(-1, kvCache.getMaxSequencePosition(0));
    }
    
    @Test
    @DisplayName("Should shift context left and keep specified tokens")
    void shouldShiftContextLeft() {
        prefill();
        int keep = 2;
        int removed = kvCache.shiftContextLeft(keep);
        assertEquals(promptTokens.length - keep, removed);
        assertEquals(keep - 1, kvCache.getMaxSequencePosition(0));
    }
}
