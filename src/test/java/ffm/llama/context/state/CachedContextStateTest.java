package ffm.llama.context.state;

import ffm.llama.config.ModelConfig;
import ffm.llama.context.state.CachedContextState;
import ffm.llama.enums.KVCacheType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CachedContextStateTest {
    
    private static final String TEST_MODEL_PATH = "/path/to/model.gguf";
    private static final byte[] DUMMY_BYTES = new byte[1024 * 1024]; // 1 MB
    private ModelConfig baseConfig;
    
    @BeforeEach
    void setUp() {
        baseConfig = ModelConfig.Builder.create()
                .contextSize(4096)
                .batchSize(512)
                .cacheTypeK(KVCacheType.Q8_0)
                .cacheTypeV(KVCacheType.TQ3_0)
                .embeddings(false)
                .build();
    }

    @Test
    @DisplayName("toString should contain path, size, and age placeholders")
    void toStringShouldContainEssentialFields() {
        CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, 0, TEST_MODEL_PATH, 0);
        String s = state.toString();
        assertTrue(s.contains("CachedState["));
        assertTrue(s.contains("path=" + TEST_MODEL_PATH));
        assertTrue(s.contains("size="));
        assertTrue(s.contains("age="));
        assertTrue(s.contains("MB"));
        assertTrue(s.contains("ms"));
    }

    @Nested
    @DisplayName("isCompatibleWith")
    class IsCompatibleWith {

        @Test
        @DisplayName("Should return true when all criteria match")
        void shouldReturnTrueWhenAllMatch() {
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, 0, TEST_MODEL_PATH, 0);
            assertTrue(state.isCompatibleWith(TEST_MODEL_PATH, baseConfig));
        }

        @Test
        @DisplayName("Should return false when model path differs")
        void shouldRejectDifferentPath() {
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, 0, TEST_MODEL_PATH, 0);
            assertFalse(state.isCompatibleWith("/other/path.gguf", baseConfig));
        }

        @Test
        @DisplayName("Should return false when context size differs")
        void shouldRejectDifferentContextSize() {
            ModelConfig other = ModelConfig.Builder.create()
                    .contextSize(2048)  // different from 4096
                    .batchSize(512)
                    .cacheTypeK(KVCacheType.Q8_0)
                    .cacheTypeV(KVCacheType.TQ3_0)
                    .embeddings(false)
                    .build();
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, 0, TEST_MODEL_PATH, 0);
            assertFalse(state.isCompatibleWith(TEST_MODEL_PATH, other));
        }

        @Test
        @DisplayName("Should return false when batch size differs")
        void shouldRejectDifferentBatchSize() {
            ModelConfig other = ModelConfig.Builder.create()
                    .contextSize(4096)
                    .batchSize(256)    // different
                    .cacheTypeK(KVCacheType.Q8_0)
                    .cacheTypeV(KVCacheType.TQ3_0)
                    .embeddings(false)
                    .build();
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, 0, TEST_MODEL_PATH, 0);
            assertFalse(state.isCompatibleWith(TEST_MODEL_PATH, other));
        }

        @Test
        @DisplayName("Should return false when K cache type differs")
        void shouldRejectDifferentKCacheType() {
            ModelConfig other = ModelConfig.Builder.create()
                    .contextSize(4096)
                    .batchSize(512)
                    .cacheTypeK(KVCacheType.F16)   // different
                    .cacheTypeV(KVCacheType.TQ3_0)
                    .embeddings(false)
                    .build();
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, 0, TEST_MODEL_PATH, 0);
            assertFalse(state.isCompatibleWith(TEST_MODEL_PATH, other));
        }

        @Test
        @DisplayName("Should return false when V cache type differs")
        void shouldRejectDifferentVCacheType() {
            ModelConfig other = ModelConfig.Builder.create()
                    .contextSize(4096)
                    .batchSize(512)
                    .cacheTypeK(KVCacheType.Q8_0)
                    .cacheTypeV(KVCacheType.F16)   // different
                    .embeddings(false)
                    .build();
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, 0, TEST_MODEL_PATH, 0);
            assertFalse(state.isCompatibleWith(TEST_MODEL_PATH, other));
        }

        @Test
        @DisplayName("Should return false when embedding mode differs")
        void shouldRejectDifferentEmbeddingMode() {
            ModelConfig other = ModelConfig.Builder.create()
                    .contextSize(4096)
                    .batchSize(512)
                    .cacheTypeK(KVCacheType.Q8_0)
                    .cacheTypeV(KVCacheType.TQ3_0)
                    .embeddings(true)   // different
                    .build();
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, 0, TEST_MODEL_PATH, 0);
            assertFalse(state.isCompatibleWith(TEST_MODEL_PATH, other));
        }
    }

    @Nested
    @DisplayName("getAgeMs")
    class GetAgeMs {

        @Test
        @DisplayName("Should return positive age when saved time is in the past")
        void shouldReturnPositiveAge() {
            long past = System.currentTimeMillis() - 10_000;
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, past, TEST_MODEL_PATH, 0);
            long age = state.getAgeMs();
            assertTrue(age >= 10_000, "Age should be at least 10 seconds");
            // Allow some tolerance for execution time
            assertTrue(age < 20_000, "Age should not be wildly inflated");
        }

        @Test
        @DisplayName("Should be zero when saved just now")
        void shouldBeZeroForCurrentTime() {
            long now = System.currentTimeMillis();
            CachedContextState state = new CachedContextState(DUMMY_BYTES, baseConfig, now, TEST_MODEL_PATH, 0);
            long age = state.getAgeMs();
            assertTrue(age >= 0);
            assertTrue(age < 1_000); // within 1 second
        }
    }

    @Nested
    @DisplayName("getSizeMB")
    class GetSizeMB {

        @Test
        @DisplayName("Should return 0 for empty byte array")
        void shouldReturnZeroForEmpty() {
            CachedContextState state = new CachedContextState(new byte[0], baseConfig, 0, TEST_MODEL_PATH, 0);
            assertEquals(0.0, state.getSizeMB(), 0.001);
        }

        @Test
        @DisplayName("Should return roughly 1.0 for 1 MB")
        void shouldReturnOneForOneMegabyte() {
            CachedContextState state = new CachedContextState(new byte[1_048_576], baseConfig, 0, TEST_MODEL_PATH, 0);
            assertEquals(1.0, state.getSizeMB(), 0.01);
        }
    }
}