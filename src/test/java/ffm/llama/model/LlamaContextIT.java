package ffm.llama.model;

import ffm.llama.IntegrationTestBase;
import ffm.llama.config.ModelConfig;

import org.junit.jupiter.api.*;

import java.nio.file.Path;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class LlamaContextIT extends IntegrationTestBase {
    
    private static LlamaModel sharedModel;
    private LlamaContext context;
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
    
    @Test
    @DisplayName("Should create context and return correct model and config")
    void shouldCreateContext() {
        assertNotNull(context);
        assertNotNull(context.ptr());
        assertSame(sharedModel, context.getModel());
        assertEquals(config, context.getModelConfig());
    }
    
    @Nested
    @DisplayName("KV cache operations")
    class KvCache {
        
        @BeforeEach
        void prefill() {
            LlamaBatch batch = LlamaBatch.create(promptTokens.length);
            for (int i = 0; i < promptTokens.length; i++) {
                batch.add(promptTokens[i], i, 0, i == promptTokens.length - 1);
            }
            assertEquals(0, context.decode(batch));
        }
        
        @Test
        @DisplayName("Should report correct max sequence position after prefill")
        void shouldReportMaxPosition() {
            assertEquals(promptTokens.length - 1, context.getMaxSequencePosition(0));
        }
        
        @Test
        @DisplayName("Should clear KV cache and reset position")
        void shouldClearKvCache() {
            context.clearKvCache();
            assertEquals(-1, context.getMaxSequencePosition(0));
        }
        
        @Test
        @DisplayName("Should shift context left and keep specified tokens")
        void shouldShiftContextLeft() {
            int keep = 2;
            int removed = context.shiftContextLeft(keep);
            assertEquals(promptTokens.length - keep, removed);
            assertEquals(keep - 1, context.getMaxSequencePosition(0));
        }
        
    }
    
    @Nested
    @DisplayName("State save & load")
    class State {
        
        private long stateSize;
        
        @BeforeEach
        void prefillAndGetSize() {
            LlamaBatch batch = LlamaBatch.create(promptTokens.length);
            for (int i = 0; i < promptTokens.length; i++) {
                batch.add(promptTokens[i], i, 0, i == promptTokens.length - 1);
            }
            context.decode(batch);
            stateSize = context.getStateSize();
            assumeTrue(stateSize > 0, "State size must be positive after prefill");
        }
        
        @Test
        @DisplayName("Should return positive state size after prefill")
        void shouldHavePositiveStateSize() {
            assertTrue(stateSize > 0);
        }
        
        @Test
        @DisplayName("Should save and load state successfully")
        void shouldSaveAndLoadState() {
            byte[] saved;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment buffer = arena.allocate(stateSize);
                long written = context.saveState(buffer, stateSize);
                assertEquals(stateSize, written, "saveState should write exactly stateSize bytes");
                saved = buffer.asSlice(0, stateSize).toArray(ValueLayout.JAVA_BYTE);
            }
            
            context.clearKvCache();
            assertEquals(-1, context.getMaxSequencePosition(0));
            
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment buffer = arena.allocateFrom(ValueLayout.JAVA_BYTE, saved);
                long read = context.loadState(buffer, stateSize);
                assertEquals(stateSize, read, "loadState should read exactly stateSize bytes");
            }
            
            assertEquals(promptTokens.length - 1, context.getMaxSequencePosition(0));
        }
    }
    
    @Test
    @DisplayName("Should estimate KV cache usage in GB")
    void shouldEstimateKvCacheUsageGB() {
        double usage = context.estimateKvCacheUsageGB();
        assertTrue(usage >= 0.0);
    }
    
    @Test
    @DisplayName("Should print info without errors")
    void shouldPrintInfo() {
        assertDoesNotThrow(() -> context.printInfo());
    }
    
    @Test
    @DisplayName("Should print performance stats without errors")
    void shouldPrintPerfStats() {
        assertDoesNotThrow(() -> context.printPerformanceStats());
    }
    
    @Test
    @DisplayName("Should reset performance stats without errors")
    void shouldResetPerfStats() {
        assertDoesNotThrow(() -> context.resetPerformanceStats());
    }
    
    @Test
    @DisplayName("Should close without errors")
    void shouldCloseCleanly() {
        assertDoesNotThrow(() -> context.close());
        context = null; // avoid double close in tearDown
    }
}