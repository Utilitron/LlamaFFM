package ffm.llama.model.state;

import ffm.llama.IntegrationTestBase;
import ffm.llama.config.ModelConfig;
import ffm.llama.model.LlamaBatch;
import ffm.llama.model.LlamaContext;
import ffm.llama.model.LlamaModel;
import ffm.llama.service.LlmService;
import org.junit.jupiter.api.*;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class ContextStateManagerIT extends IntegrationTestBase {
    
    private static final String MODEL_RESOURCE_ENV = "LLAMA_TEST_MODEL_RESOURCE";
    
    private LlamaModel model;
    private LlamaContext context;
    private ModelConfig config;
    
    @BeforeEach
    void setUp() {
        // e.g., export LLAMA_TEST_MODEL_RESOURCE=models/tiny_q4_0.gguf
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
    }
    
    @AfterEach
    void tearDown() {
        if (context != null) context.close();
        if (model != null) model.close();
    }
    
    @Test
    @DisplayName("Should snapshot and restore context successfully")
    void shouldSnapshotAndRestoreContext() {
        // Prefill a few tokens
        int[] tokens = {1, 2, 3, 4};
        context.decode(LlamaBatch.create(tokens.length)); // simplified; real code uses GenerationSession
        int posBefore = context.getMaxSequencePosition(0) + 1;
        
        // Snapshot
        var state = ContextStateManager.snapshotContext(
                new LlmService.ModelInstance(model, context, config)
        ).orElseThrow();
        assertNotNull(state);
        assertEquals(posBefore, state.getNTokens());
        
        // Clear context
        context.clearKvCache();
        assertEquals(0, context.getMaxSequencePosition(0) + 1);
        
        // Restore
        boolean restored = ContextStateManager.restoreContext(context, state);
        assertTrue(restored);
        assertEquals(posBefore, context.getMaxSequencePosition(0) + 1);
    }
}
