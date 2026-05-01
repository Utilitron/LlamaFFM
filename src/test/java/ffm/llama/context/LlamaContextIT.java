package ffm.llama.context;

import ffm.llama.IntegrationTestBase;
import ffm.llama.batch.LlamaBatch;
import ffm.llama.config.ModelConfig;

import ffm.llama.model.LlamaModel;
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
    
    @Test
    @DisplayName("Should estimate KV cache usage in GB")
    void shouldEstimateKvCacheUsageGB() {
        double usage = context.kvCache().estimateKvCacheUsageGB();
        assertTrue(usage >= 0.0);
    }
    
    @Test
    @DisplayName("Should print info without errors")
    void shouldPrintInfo() {
        assertDoesNotThrow(() -> context.printInfo());
    }
    
    @Test
    @DisplayName("Should close without errors")
    void shouldCloseCleanly() {
        assertDoesNotThrow(() -> context.close());
        context = null; // avoid double close in tearDown
    }
}