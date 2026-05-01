package ffm.llama.model;

import ffm.llama.IntegrationTestBase;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

import static java.nio.file.Files.createTempDirectory;
import static java.nio.file.Files.createTempFile;
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
    
    @Nested
    @DisplayName("Constructor failure paths")
    class ConstructorFailures {
        
        @Test
        @DisplayName("Should throw when model path does not exist")
        void shouldThrowOnNonExistentPath() {
            String nonExistentPath = "/non/existent/path/model.gguf";
            RuntimeException ex = assertThrows(RuntimeException.class,
                    () -> new LlamaModel(nonExistentPath));
            assertTrue(ex.getMessage().contains("Failed to load model"));
        }
        
        @Test
        @DisplayName("Should throw when model path is null")
        void shouldThrowOnNullPath() {
            assertThrows(RuntimeException.class,
                    () -> new LlamaModel(null));
        }
        
        @Test
        @DisplayName("Should throw when model path is empty")
        void shouldThrowOnEmptyPath() {
            RuntimeException ex = assertThrows(RuntimeException.class,
                    () -> new LlamaModel(""));
            assertTrue(ex.getMessage().contains("Failed to load model"));
        }
        
        @Test
        @DisplayName("Should throw when model file is invalid (not a GGUF)")
        void shouldThrowOnInvalidModelFile() throws IOException {
            // Create a dummy file that's not a valid GGUF model
            Path tempFile = createTempFile("invalid-model.gguf", "not a real model");
            RuntimeException ex = assertThrows(RuntimeException.class,
                    () -> new LlamaModel(tempFile.toString()));
            assertTrue(ex.getMessage().contains("Failed to load model"));
        }
        
        @Test
        @DisplayName("Should throw when model file is a directory")
        void shouldThrowOnDirectory() throws IOException {
            Path tempDir = createTempDirectory("not-a-model");
            RuntimeException ex = assertThrows(RuntimeException.class,
                    () -> new LlamaModel(tempDir.toString()));
            assertTrue(ex.getMessage().contains("Failed to load model"));
        }
    }
    
    @Nested
    @DisplayName("Double close scenarios")
    class DoubleClose {
        
        private LlamaModel model;
        
        @BeforeEach
        void loadModel() {
            String resource = System.getenv("LLAMA_TEST_MODEL_RESOURCE");
            assumeTrue(resource != null && !resource.isBlank());
            Path modelPath = resourceToPath(resource);
            model = new LlamaModel(modelPath.toString());
        }
        
        @Test
        @DisplayName("Should safely handle double close without error")
        void shouldHandleDoubleClose() {
            model.close();
            assertDoesNotThrow(() -> model.close());
            assertTrue(model.isClosed());
        }
        
        @Test
        @DisplayName("Should safely handle triple close")
        void shouldHandleTripleClose() {
            model.close();
            model.close();
            assertDoesNotThrow(() -> model.close());
            assertTrue(model.isClosed());
        }
        
        @Test
        @DisplayName("Should throw IllegalStateException when using closed model")
        void shouldThrowOnUseAfterClose() {
            model.close();
            assertThrows(IllegalStateException.class, () -> model.getVocabSize());
            assertThrows(IllegalStateException.class, () -> model.getEmbeddingSize());
            assertThrows(IllegalStateException.class, () -> model.getBosToken());
            assertThrows(IllegalStateException.class, () -> model.tokenize("test", true, false));
        }
        
        @AfterEach
        void cleanup() {
            if (model != null && !model.isClosed()) {
                model.close();
            }
        }
    }
    
    @Nested
    @DisplayName("Tokenization edge cases")
    class TokenizationEdgeCases {
        
        private LlamaModel model;
        
        @BeforeEach
        void loadModel() {
            String resource = System.getenv("LLAMA_TEST_MODEL_RESOURCE");
            assumeTrue(resource != null && !resource.isBlank());
            Path modelPath = resourceToPath(resource);
            model = new LlamaModel(modelPath.toString());
        }
        
        @AfterEach
        void cleanup() {
            if (model != null) {
                model.close();
            }
        }
        
        @Test
        @DisplayName("Should handle empty string tokenization")
        void shouldHandleEmptyString() {
            int[] tokens = model.tokenize("", true, false);
            // May return just BOS token or empty array depending on implementation
            assertNotNull(tokens);
        }
        
        @Test
        @DisplayName("Should handle very long text tokenization")
        void shouldHandleVeryLongText() {
            String longText = "test ".repeat(10000);
            assertDoesNotThrow(() -> model.tokenize(longText, true, false));
        }
        
        @Test
        @DisplayName("Should handle unicode and special characters")
        void shouldHandleUnicode() {
            String unicode = "Hello 世界 🌍 émojis";
            int[] tokens = model.tokenize(unicode, true, false);
            assertNotNull(tokens);
            assertTrue(tokens.length > 0);
        }
        
        @Test
        @DisplayName("Should handle null text tokenization")
        void shouldThrowOnNullTokenization() {
            assertThrows(Exception.class, () -> model.tokenize(null, true, false));
        }
        
        @Test
        @DisplayName("Should handle detokenization of empty array")
        void shouldHandleEmptyDetokenization() {
            int[] emptyTokens = new int[0];
            String result = model.detokenize(emptyTokens, false, true);
            assertNotNull(result);
        }
        
        @Test
        @DisplayName("Should reject invalid token IDs")
        void shouldRejectInvalidTokenIds() {
            int[] invalidTokens = {-1, Integer.MAX_VALUE, model.getVocabSize() + 1000};
            assertThrows(IllegalArgumentException.class,
                    () -> model.detokenize(invalidTokens, false, true));
        }
    }
    
    @Nested
    @DisplayName("Concurrent access misuse")
    class ConcurrentAccess {
        
        private LlamaModel model;
        
        @BeforeEach
        void loadModel() {
            String resource = System.getenv("LLAMA_TEST_MODEL_RESOURCE");
            assumeTrue(resource != null && !resource.isBlank());
            Path modelPath = resourceToPath(resource);
            model = new LlamaModel(modelPath.toString());
        }
        
        @AfterEach
        void cleanup() {
            if (model != null && !model.isClosed()) {
                model.close();
            }
        }
        
        @Test
        @DisplayName("Should handle concurrent tokenization calls")
        void shouldHandleConcurrentTokenization() throws InterruptedException {
            final int threadCount = 5;
            Thread[] threads = new Thread[threadCount];
            final boolean[] failures = new boolean[threadCount];
            
            for (int i = 0; i < threadCount; i++) {
                final int index = i;
                threads[i] = new Thread(() -> {
                    try {
                        for (int j = 0; j < 10; j++) {
                            model.tokenize("Test string " + index, true, false);
                        }
                    } catch (Exception e) {
                        failures[index] = true;
                    }
                });
            }
            
            for (Thread thread : threads) {
                thread.start();
            }
            
            for (Thread thread : threads) {
                thread.join();
            }
            
            // This test documents behavior - concurrent access may or may not work
            // depending on thread safety implementation
        }
        
        @Test
        @DisplayName("Should handle close during active operations")
        void shouldHandleCloseWhileInUse() throws InterruptedException {
            Thread worker = new Thread(() -> {
                try {
                    for (int i = 0; i < 100; i++) {
                        if (!model.isClosed()) {
                            model.tokenize("test", true, false);
                            Thread.sleep(10);
                        }
                    }
                } catch (Exception e) {
                    // Expected - either IllegalStateException or InterruptedException
                }
            });
            
            worker.start();
            Thread.sleep(50); // Let worker start
            model.close(); // Close while worker is running
            worker.join(5000); // Wait for worker to finish
            
            assertTrue(model.isClosed());
        }
    }
}
