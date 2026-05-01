package ffm.llama.utils;

import ffm.llama.IntegrationTestBase;
import ffm.llama.config.ModelConfig;
import ffm.llama.context.LlamaContext;
import ffm.llama.model.LlamaModel;
import org.junit.jupiter.api.*;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class PerformanceMonitorIT extends IntegrationTestBase {
    
    private static LlamaModel sharedModel;
    private LlamaContext context;
    private PerformanceMonitor monitor;
    
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
        // Create a real context so the native pointer is valid
        context = new LlamaContext(sharedModel,
                ModelConfig.Builder.create()
                        .contextSize(64)
                        .batchSize(16)
                        .cpuThreads(1)
                        .embeddings(false)
                        .build());
        
        // Instantiate the PerformanceMonitor with the context's native pointer
        monitor = new PerformanceMonitor(context.ptr());
    }
    
    @AfterEach
    void tearDown() {
        if (context != null) {
            context.close();
            context = null;
        }
    }
    
    @Test
    @DisplayName("Should print performance stats without errors")
    void shouldPrintPerfStats() {
        assertDoesNotThrow(() -> monitor.printPerformanceStats());
    }
    
    @Test
    @DisplayName("Should reset performance stats without errors")
    void shouldResetPerfStats() {
        assertDoesNotThrow(() -> monitor.resetPerformanceStats());
    }
}
