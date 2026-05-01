package ffm.llama.context.state;

import ffm.llama.IntegrationTestBase;
import ffm.llama.batch.BatchDecoder;
import ffm.llama.batch.LlamaBatch;
import ffm.llama.cache.KvCacheManager;
import ffm.llama.config.ModelConfig;
import ffm.llama.context.LlamaContext;
import ffm.llama.model.LlamaModel;
import org.junit.jupiter.api.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class ContextStateIT extends IntegrationTestBase {
    
    private static LlamaModel sharedModel;
    private LlamaContext context;
    private ContextStateIO stateIO;
    private KvCacheManager kvCache;
    private BatchDecoder decoder;
    private ModelConfig config;
    private int[] promptTokens;
    private long stateSize;
    
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
        
        // Create helpers using the context's native pointer
        MemorySegment ctxPtr = context.ptr();
        stateIO = new ContextStateIO(ctxPtr);
        kvCache = new KvCacheManager(ctxPtr, sharedModel);
        decoder  = new BatchDecoder(ctxPtr);
        
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
    
    private void prefill() {
        LlamaBatch batch = LlamaBatch.create(promptTokens.length);
        for (int i = 0; i < promptTokens.length; i++) {
            batch.add(promptTokens[i], i, 0, i == promptTokens.length - 1);
        }
        assertEquals(0, decoder.decode(batch));
        stateSize = stateIO.getStateSize();
        assumeTrue(stateSize > 0, "State size must be positive after prefill");
    }
    
    @Test
    @DisplayName("Should return positive state size after prefill")
    void shouldHavePositiveStateSize() {
        prefill();
        assertTrue(stateSize > 0);
    }
    
    @Test
    @DisplayName("Should save and load state successfully")
    void shouldSaveAndLoadState() {
        prefill();
        byte[] saved;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buffer = arena.allocate(stateSize);
            long written = stateIO.saveState(buffer, stateSize);
            assertEquals(stateSize, written, "saveState should write exactly stateSize bytes");
            saved = buffer.asSlice(0, stateSize).toArray(ValueLayout.JAVA_BYTE);
        }
        
        kvCache.clearKvCache();
        assertEquals(-1, kvCache.getMaxSequencePosition(0));
        
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buffer = arena.allocateFrom(ValueLayout.JAVA_BYTE, saved);
            long read = stateIO.loadState(buffer, stateSize);
            assertEquals(stateSize, read, "loadState should read exactly stateSize bytes");
        }
        
        assertEquals(promptTokens.length - 1, kvCache.getMaxSequencePosition(0));
    }
}