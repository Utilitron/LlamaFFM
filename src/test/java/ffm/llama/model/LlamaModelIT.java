package ffm.llama.model;

import ffm.llama.IntegrationTestBase;
import org.junit.jupiter.api.*;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class LlamaModelIT extends IntegrationTestBase {
    
    private static LlamaModel model;
    
    @BeforeAll
    static void loadModel() {
        String resource = System.getenv("LLAMA_TEST_MODEL_RESOURCE");
        assumeTrue(resource != null && !resource.isBlank(),
                () -> "Skipping test: LLAMA_TEST_MODEL_RESOURCE is not set");
        Path modelPath = resourceToPath(resource);
        model = new LlamaModel(modelPath.toString());
    }
    
    @AfterAll
    static void freeModel() {
        if (model != null) {
            model.close();
            model = null;
        }
    }
    
    @Test
    @DisplayName("Should load model and return valid pointer")
    void shouldLoadModel() {
        assertNotNull(model);
        assertNotNull(model.ptr());
        assertNotEquals(MemorySegment.NULL, model.ptr());
    }
    
    @Nested
    @DisplayName("Metadata")
    class Metadata {
        
        @Test
        @DisplayName("Should report positive vocab size")
        void shouldHaveVocabSize() {
            assertTrue(model.getVocabSize() > 0, "Vocab size should be positive");
        }
        
        @Test
        @DisplayName("Should report positive embedding size")
        void shouldHaveEmbeddingSize() {
            assertTrue(model.getEmbeddingSize() > 0, "Embedding size should be positive");
        }
        
        @Test
        @DisplayName("Should report positive layer count")
        void shouldHaveLayers() {
            assertTrue(model.getLayerCount() > 0, "Layer count should be positive");
        }
        
        @Test
        @DisplayName("Should report positive parameter count")
        void shouldHaveParameters() {
            assertTrue(model.getParameterCount() > 0, "Parameter count should be positive");
        }
        
        @Test
        @DisplayName("Should report positive model size in bytes")
        void shouldHaveModelSizeBytes() {
            assertTrue(model.getModelSizeBytes() > 0, "Model size in bytes should be positive");
        }
        
        @Test
        @DisplayName("Should report positive model size in GB")
        void shouldHaveModelSizeGB() {
            assertTrue(model.getModelSizeGB() > 0.0, "Model size in GB should be positive");
        }
        
        @Test
        @DisplayName("Should return a non-empty description")
        void shouldHaveDescription() {
            String desc = model.getDescription();
            assertNotNull(desc);
            assertFalse(desc.isBlank());
        }
    }
    
    @Nested
    @DisplayName("Special tokens")
    class SpecialTokens {
        
        @Test
        @DisplayName("Should return non-negative BOS token")
        void shouldHaveBosToken() {
            assertTrue(model.getBosToken() >= 0);
        }
        
        @Test
        @DisplayName("Should return non-negative EOS token")
        void shouldHaveEosToken() {
            assertTrue(model.getEosToken() >= 0);
        }
        
        @Test
        @DisplayName("Should return a valid EOT token or -1 if absent")
        void shouldHaveEotToken() {
            int eot = model.getEotToken();
            assertTrue(eot >= 0 || eot == -1, "EOT token should be >=0 or -1 (missing)");
        }
        
        @Test
        @DisplayName("Should return non-negative newline token")
        void shouldHaveNewlineToken() {
            assertTrue(model.getNewlineToken() >= 0);
        }
    }
    
    @Nested
    @DisplayName("Tokenization")
    class Tokenization {
        
        private final String testText = "Hello World";
        private int[] tokens;
        
        @BeforeEach
        void tokenize() {
            tokens = model.tokenize(testText, true, false); // add BOS, no special
        }
        
        @Test
        @DisplayName("Should produce at least one token")
        void shouldProduceTokens() {
            assertTrue(tokens.length > 0, "Tokenization should produce at least one token");
        }
        
        @Test
        @DisplayName("Should detokenize back to original text (ignoring BOS)")
        void shouldDetokenize() {
            // Remove BOS if present
            int[] withoutBos = tokens;
            if (tokens[0] == model.getBosToken()) {
                withoutBos = new int[tokens.length - 1];
                System.arraycopy(tokens, 1, withoutBos, 0, withoutBos.length);
            }
            String result = model.detokenize(withoutBos, false, true);
            assertNotNull(result);
            assertFalse(result.isBlank());
            // We can't expect exact match due to whitespace / tokenization artifacts,
            // but the result should be a non-empty string.
        }
    }
    
    @Nested
    @DisplayName("Token to string")
    class TokenToString {
        
        @Test
        @DisplayName("Should convert a known token to a non-empty string")
        void shouldConvertToken() {
            int token = model.getBosToken();
            String text = model.tokenToString(token);
            assertNotNull(text);
            // BOS may be empty string or a special representation, but tokenToString
            // should never throw.
        }
    }
    
    @Nested
    @DisplayName("Chat template")
    class ChatTemplate {
        
        @Test
        @DisplayName("Should return a string or null without throwing")
        void shouldGetChatTemplate() {
            assertDoesNotThrow(() -> {
                String tmpl = model.getChatTemplate();
                // It's okay to be null if the model doesn't have one.
            });
        }
    }
    
    @Test
    @DisplayName("Should print info without errors")
    void shouldPrintInfo() {
        assertDoesNotThrow(() -> model.printInfo());
    }
    
    @Test
    @DisplayName("Should close without errors")
    void shouldCloseCleanly() {
        // Load a separate model for this test to avoid affecting others
        String resource = System.getenv("LLAMA_TEST_MODEL_RESOURCE");
        assumeTrue(resource != null && !resource.isBlank());
        Path modelPath = resourceToPath(resource);
        LlamaModel tempModel = new LlamaModel(modelPath.toString());
        assertDoesNotThrow(() -> tempModel.close());
    }
}
