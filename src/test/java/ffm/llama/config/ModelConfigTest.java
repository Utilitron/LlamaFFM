package ffm.llama.config;

import ffm.llama.enums.KVCacheType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link ModelConfig}.
 * Covers builder defaults, field propagation, validation, factory presets,
 * immutability, and the {@code estimateKvCacheSizeGB} utility.
 */
class ModelConfigTest {

    @Nested
    @DisplayName("Builder – default values")
    class BuilderDefaults {

        @Test
        @DisplayName("Should use documented defaults when no setters are called")
        void shouldProvideDocumentedDefaults() {
            ModelConfig cfg = ModelConfig.Builder.create().build();

            assertEquals(0, cfg.getGpuLayers());
            assertFalse(cfg.isOffloadKvToGpu());
            assertTrue(cfg.isUseMmap());
            assertFalse(cfg.isUseMlock());
            assertEquals(2048, cfg.getContextSize());
            assertEquals(512, cfg.getBatchSize());
            assertTrue(cfg.getCpuThreads() > 0, "CPU threads should be positive");
            assertEquals(0.1f, cfg.getDefragThreshold(), 0.0f);
            assertFalse(cfg.isFlashAttention());
            assertFalse(cfg.isEmbeddings());
            assertEquals(KVCacheType.F16, cfg.getCacheTypeK());
            assertEquals(KVCacheType.F16, cfg.getCacheTypeV());
            assertNull(cfg.getJsonSchema());
            assertFalse(cfg.isEnableGrammar());
            assertFalse(cfg.isDynamicAttentionSharpening());
            assertEquals(0.1, cfg.getAttentionSharpeningFactor(), 0.0);
        }
    }

    @Nested
    @DisplayName("Builder – field propagation")
    class BuilderFieldPropagation {

        @Test
        @DisplayName("Should set every field correctly via builder")
        void shouldSetAllFields() {
            ModelConfig cfg = ModelConfig.Builder.create()
                    .gpuLayers(33)
                    .offloadKvToGpu(true)
                    .useMmap(false)
                    .useMlock(true)
                    .contextSize(8192)
                    .batchSize(256)
                    .cpuThreads(8)
                    .defragThreshold(0.25f)
                    .flashAttention(true)
                    .embeddings(true)
                    .cacheTypeK(KVCacheType.Q8_0)
                    .cacheTypeV(KVCacheType.TQ3_0)
                    .jsonSchema("{}")
                    .dynamicAttentionSharpening(true)
                    .attentionSharpeningFactor(0.5)
                    .build();

            assertEquals(33, cfg.getGpuLayers());
            assertTrue(cfg.isOffloadKvToGpu());
            assertFalse(cfg.isUseMmap());
            assertTrue(cfg.isUseMlock());
            assertEquals(8192, cfg.getContextSize());
            assertEquals(256, cfg.getBatchSize());
            assertEquals(8, cfg.getCpuThreads());
            assertEquals(0.25f, cfg.getDefragThreshold(), 0.0f);
            assertTrue(cfg.isFlashAttention());
            assertTrue(cfg.isEmbeddings());
            assertEquals(KVCacheType.Q8_0, cfg.getCacheTypeK());
            assertEquals(KVCacheType.TQ3_0, cfg.getCacheTypeV());
            assertEquals("{}", cfg.getJsonSchema());
            assertTrue(cfg.isEnableGrammar());
            assertTrue(cfg.isDynamicAttentionSharpening());
            assertEquals(0.5, cfg.getAttentionSharpeningFactor(), 0.0);
        }

        @Test
        @DisplayName("Partial builder usage should not alter untouched fields")
        void partialBuilderUsageShouldRetainDefaults() {
            ModelConfig cfg = ModelConfig.Builder.create()
                    .cacheTypeK(KVCacheType.TQ3_0)
                    .build();

            // Only the explicitly set field should change
            assertEquals(KVCacheType.TQ3_0, cfg.getCacheTypeK());
            assertEquals(KVCacheType.F16, cfg.getCacheTypeV(),
                    "cacheTypeV should remain default when only cacheTypeK is set");
        }
    }

    @Nested
    @DisplayName("Builder – precedence and overrides")
    class BuilderPrecedence {

        @Test
        @DisplayName("Explicit enableGrammar(false) should override jsonSchema auto-enable")
        void explicitGrammarShouldOverrideSchemaAutoEnable() {
            ModelConfig cfg = ModelConfig.Builder.create()
                    .jsonSchema("{\"type\":\"object\"}")
                    .enableGrammar(false)
                    .build();

            assertFalse(cfg.isEnableGrammar(),
                    "Explicit enableGrammar(false) must take precedence");
            assertEquals("{\"type\":\"object\"}", cfg.getJsonSchema());
        }

        @Test
        @DisplayName("Later jsonSchema(null) should disable auto-enabled grammar")
        void nullSchemaShouldDisableGrammarAfterAutoEnable() {
            ModelConfig cfg = ModelConfig.Builder.create()
                    .jsonSchema("{}")      // enables grammar
                    .jsonSchema(null)      // should disable
                    .build();

            assertFalse(cfg.isEnableGrammar());
            assertNull(cfg.getJsonSchema());
        }
    }

    @Nested
    @DisplayName("Builder – validation (current behaviour and documented gaps)")
    class BuilderValidation {

        @Test
        @DisplayName("Should reject sharpening factor < 0")
        void shouldRejectNegativeSharpeningFactor() {
            assertThrows(IllegalArgumentException.class,
                    () -> ModelConfig.Builder.create()
                            .attentionSharpeningFactor(-0.1)
                            .build());
        }

        @Test
        @DisplayName("Should reject sharpening factor > 1.0")
        void shouldRejectTooLargeSharpeningFactor() {
            assertThrows(IllegalArgumentException.class,
                    () -> ModelConfig.Builder.create()
                            .attentionSharpeningFactor(1.0001)
                            .build());
        }

        @Test
        @DisplayName("Should accept sharpening factor exactly at bounds")
        void shouldAcceptBoundarySharpeningFactor() {
            ModelConfig minCfg = ModelConfig.Builder.create()
                    .attentionSharpeningFactor(0.0).build();
            assertEquals(0.0, minCfg.getAttentionSharpeningFactor());

            ModelConfig maxCfg = ModelConfig.Builder.create()
                    .attentionSharpeningFactor(1.0).build();
            assertEquals(1.0, maxCfg.getAttentionSharpeningFactor());
        }

        // --- Fields that currently accept any value (gaps in validation) ---
        // These tests document the *current* permissive behaviour.
        // If validation is added later, these tests will fail and serve as
        // regression checks for the new, stricter contract.

        @Test
        @DisplayName("Currently accepts non-positive context size (validation gap)")
        void currentlyAcceptsZeroContextSize() {
            ModelConfig cfg = ModelConfig.Builder.create().contextSize(0).build();
            assertEquals(0, cfg.getContextSize());
        }

        @Test
        @DisplayName("Currently accepts negative batch size (validation gap)")
        void currentlyAcceptsNegativeBatchSize() {
            ModelConfig cfg = ModelConfig.Builder.create().batchSize(-1).build();
            assertEquals(-1, cfg.getBatchSize());
        }

        @Test
        @DisplayName("Currently accepts negative CPU threads (validation gap)")
        void currentlyAcceptsNegativeCpuThreads() {
            ModelConfig cfg = ModelConfig.Builder.create().cpuThreads(-4).build();
            assertEquals(-4, cfg.getCpuThreads());
        }

        @Test
        @DisplayName("Currently accepts negative GPU layers (validation gap)")
        void currentlyAcceptsNegativeGpuLayers() {
            ModelConfig cfg = ModelConfig.Builder.create().gpuLayers(-5).build();
            assertEquals(-5, cfg.getGpuLayers());
        }

        @Test
        @DisplayName("Currently accepts negative defrag threshold (validation gap)")
        void currentlyAcceptsNegativeDefragThreshold() {
            ModelConfig cfg = ModelConfig.Builder.create().defragThreshold(-0.5f).build();
            assertEquals(-0.5f, cfg.getDefragThreshold(), 0.0f);
        }
    }

    @Nested
    @DisplayName("Builder – null handling for enums")
    class NullEnumHandling {

        @Test
        @DisplayName("Null cache type should be allowed (if no validation)")
        void nullCacheTypeShouldBeStoredAsNull() {
            // Note: This behaviour can cause NullPointerException later in native code.
            // The test documents the current contract so that a decision can be made.
            ModelConfig cfg = ModelConfig.Builder.create()
                    .cacheTypeK(null)
                    .build();
            assertNull(cfg.getCacheTypeK());
        }
    }

    @Nested
    @DisplayName("Factory presets")
    class FactoryPresets {

        @Test
        @DisplayName("createDefault() should match full expected profile")
        void shouldCreateDefaultConfig() {
            ModelConfig cfg = ModelConfig.createDefault();
            assertEquals(0, cfg.getGpuLayers());
            assertFalse(cfg.isOffloadKvToGpu());
            assertTrue(cfg.isUseMmap());
            assertFalse(cfg.isUseMlock());
            assertEquals(2048, cfg.getContextSize());
            assertEquals(512, cfg.getBatchSize());
            assertTrue(cfg.getCpuThreads() > 0);
            assertEquals(0.1f, cfg.getDefragThreshold(), 0.0f);
            assertFalse(cfg.isFlashAttention());
            assertFalse(cfg.isEmbeddings());
            assertEquals(KVCacheType.F16, cfg.getCacheTypeK());
            assertEquals(KVCacheType.F16, cfg.getCacheTypeV());
            assertNull(cfg.getJsonSchema());
            assertFalse(cfg.isEnableGrammar());
            assertFalse(cfg.isDynamicAttentionSharpening());
            assertEquals(0.1, cfg.getAttentionSharpeningFactor(), 0.0);
        }

        @Test
        @DisplayName("longContextConsumer() should include full invariant set")
        void shouldCreateLongContextConsumerConfig() {
            ModelConfig cfg = ModelConfig.longContextConsumer();
            assertEquals(99, cfg.getGpuLayers());
            // offloadKvToGpu not set explicitly, check default
            assertFalse(cfg.isOffloadKvToGpu());
            assertTrue(cfg.isUseMmap());
            assertEquals(32768, cfg.getContextSize());
            assertEquals(512, cfg.getBatchSize());
            assertTrue(cfg.isFlashAttention());
            assertEquals(KVCacheType.Q8_0, cfg.getCacheTypeK());
            assertEquals(KVCacheType.TQ3_0, cfg.getCacheTypeV());
            assertTrue(cfg.isDynamicAttentionSharpening());
            assertEquals(0.1, cfg.getAttentionSharpeningFactor(), 0.0);
            // defragThreshold, cpuThreads remain defaults
            assertEquals(0.1f, cfg.getDefragThreshold(), 0.0f);
            assertTrue(cfg.getCpuThreads() > 0);
        }

        @Test
        @DisplayName("extremeCompression() should use TurboQuant and 128K context")
        void shouldCreateExtremeCompressionConfig() {
            ModelConfig cfg = ModelConfig.extremeCompression();
            assertEquals(99, cfg.getGpuLayers());
            assertTrue(cfg.isUseMmap());
            assertEquals(131072, cfg.getContextSize());
            assertEquals(512, cfg.getBatchSize());
            assertTrue(cfg.isFlashAttention());
            assertEquals(KVCacheType.TBQP3, cfg.getCacheTypeK());
            assertEquals(KVCacheType.TBQP3, cfg.getCacheTypeV());
            assertTrue(cfg.isDynamicAttentionSharpening());
            assertEquals(0.15, cfg.getAttentionSharpeningFactor(), 0.0);
        }
    }

    @Nested
    @DisplayName("Utility: estimateKvCacheSizeGB")
    class KvCacheEstimate {

        @Test
        @DisplayName("Should compute KV cache size correctly for moderate values")
        void shouldComputeKvCacheSize() {
            double expected = (2048L * 32 * 4096 * 2 * 2) / 1_000_000_000.0;
            double actual = ModelConfig.estimateKvCacheSizeGB(2048, 32, 4096);
            assertEquals(expected, actual, 0.001);
        }

        @Test
        @DisplayName("Should return 0 for zero context")
        void shouldReturnZeroForZeroContext() {
            assertEquals(0.0, ModelConfig.estimateKvCacheSizeGB(0, 32, 4096), 0.0);
        }

        @Test
        @DisplayName("Should be proportional to context size")
        void shouldBeProportionalToContext() {
            double base = ModelConfig.estimateKvCacheSizeGB(1024, 32, 4096);
            double doubled = ModelConfig.estimateKvCacheSizeGB(2048, 32, 4096);
            assertEquals(base * 2, doubled, 0.01);
        }

        @Test
        @DisplayName("Should not overflow for large 128K × 80-layer × 8192-d config")
        void shouldNotOverflowForLargeInputs() {
            double result = ModelConfig.estimateKvCacheSizeGB(131072, 80, 8192);
            assertTrue(result > 0, "Expected positive result, got " + result);
            // sanity: must be less than some absurd upper bound (e.g. 1 TB)
            assertTrue(result < 1_000_000.0);
        }
    }

    @Nested
    @DisplayName("toString representation")
    class ToString {

        @Test
        @DisplayName("Should contain all expected key fields in standard format")
        void toStringShouldContainAllKeyFields() {
            String s = ModelConfig.Builder.create()
                    .gpuLayers(5)
                    .offloadKvToGpu(true)
                    .contextSize(4096)
                    .cacheTypeK(KVCacheType.Q8_0)
                    .build()
                    .toString();

            // Checking for the standard bracketed prefix and the essential fields
            assertTrue(s.startsWith("ModelConfig["));
            assertTrue(s.contains("gpu_layers=5"));
            assertTrue(s.contains("kv_gpu=true"));
            assertTrue(s.contains("ctx=4096"));
            assertTrue(s.contains("cacheK=Q8_0"));
            assertTrue(s.endsWith("]"));
        }
    }
}
