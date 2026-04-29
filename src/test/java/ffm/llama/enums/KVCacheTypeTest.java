package ffm.llama.enums;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link KVCacheType}.
 * Verifies constant metadata, the formula inside {@code estimateVramSavingsMB},
 * and documents behavioural boundaries (including overflow/zero cases).
 */
class KVCacheTypeTest {
    
    @Nested
    @DisplayName("Constant metadata")
    class ConstantMetadata {
        
        @Test
        @DisplayName("Should have correct llamaCppName for each type")
        void shouldHaveCorrectLlamaCppName() {
            assertEquals("f16", KVCacheType.F16.getLlamaCppName());
            assertEquals("q8_0", KVCacheType.Q8_0.getLlamaCppName());
            assertEquals("q4_0", KVCacheType.Q4_0.getLlamaCppName());
            assertEquals("tq3_0", KVCacheType.TQ3_0.getLlamaCppName());
            assertEquals("tbqx3", KVCacheType.TBQX3.getLlamaCppName());
            assertEquals("tbqp3", KVCacheType.TBQP3.getLlamaCppName());
        }
        
        @Test
        @DisplayName("Should have correct nativeId for each type")
        void shouldHaveCorrectNativeId() {
            assertEquals(1, KVCacheType.F16.getNativeId());
            assertEquals(8, KVCacheType.Q8_0.getNativeId());
            assertEquals(2, KVCacheType.Q4_0.getNativeId());
            assertEquals(42, KVCacheType.TQ3_0.getNativeId());
            assertEquals(43, KVCacheType.TBQX3.getNativeId());
            assertEquals(44, KVCacheType.TBQP3.getNativeId());
        }
        
        @Test
        @DisplayName("Should have correct bitsPerWeight")
        void shouldHaveCorrectBitsPerWeight() {
            assertEquals(16, KVCacheType.F16.getBitsPerWeight(), 0.01);
            assertEquals(8, KVCacheType.Q8_0.getBitsPerWeight(), 0.01);
            assertEquals(4, KVCacheType.Q4_0.getBitsPerWeight(), 0.01);
            assertEquals(3.25, KVCacheType.TQ3_0.getBitsPerWeight(), 0.01);
            assertEquals(3.5, KVCacheType.TBQX3.getBitsPerWeight(), 0.01);
            assertEquals(3.0, KVCacheType.TBQP3.getBitsPerWeight(), 0.01);
        }
        
        @Test
        @DisplayName("Should have correct compression ratio")
        void shouldHaveCorrectCompressionRatio() {
            assertEquals(1.0, KVCacheType.F16.getCompressionRatio(), 0.01);
            assertEquals(2.0, KVCacheType.Q8_0.getCompressionRatio(), 0.01);
            assertEquals(4.0, KVCacheType.Q4_0.getCompressionRatio(), 0.01);
            assertEquals(4.9, KVCacheType.TQ3_0.getCompressionRatio(), 0.01);
            assertEquals(4.7, KVCacheType.TBQX3.getCompressionRatio(), 0.01);
            assertEquals(5.2, KVCacheType.TBQP3.getCompressionRatio(), 0.01);
        }
    }
    
    @Nested
    @DisplayName("estimateVramSavingsMB method")
    class EstimateVramSavingsMB {
        
        @Test
        @DisplayName("Should return 0 for FP16 (no savings)")
        void f16ShouldReturnZeroSavings() {
            long savings = KVCacheType.F16.estimateVramSavingsMB(32768, 8);
            assertEquals(0L, savings);
        }
        
        @Test
        @DisplayName("Should return positive savings for compressed types")
        void compressedTypesShouldReturnPositiveSavings() {
            assertTrue(KVCacheType.Q8_0.estimateVramSavingsMB(32768, 8) > 0);
            assertTrue(KVCacheType.Q4_0.estimateVramSavingsMB(32768, 8) > 0);
            assertTrue(KVCacheType.TQ3_0.estimateVramSavingsMB(32768, 8) > 0);
        }
        
        @Test
        @DisplayName("Should be monotonic: higher compression gives larger savings")
        void higherCompressionShouldGiveLargerSavings() {
            long savingsQ8 = KVCacheType.Q8_0.estimateVramSavingsMB(32768, 8);
            long savingsQ4 = KVCacheType.Q4_0.estimateVramSavingsMB(32768, 8);
            long savingsTQ = KVCacheType.TQ3_0.estimateVramSavingsMB(32768, 8);
            assertTrue(savingsQ4 > savingsQ8);
            assertTrue(savingsTQ > savingsQ4);
        }
        
        @Test
        @DisplayName("Should scale linearly with context length")
        void shouldScaleWithContextLength() {
            long base = KVCacheType.Q4_0.estimateVramSavingsMB(16384, 8);
            long scaled = KVCacheType.Q4_0.estimateVramSavingsMB(32768, 8);
            assertEquals(base * 2, scaled, base * 0.01); // within 1% of base
        }
        
        @Test
        @DisplayName("Should return 0 for zero context length")
        void zeroContextShouldReturnZero() {
            assertEquals(0L, KVCacheType.Q8_0.estimateVramSavingsMB(0, 8));
        }
        
        @Test
        @DisplayName("Should handle unknown model sizes gracefully")
        void shouldHandleUnknownModelSize() {
            // For model size not in switch, formula uses 0.5 * modelSizeB as base
            long savings = KVCacheType.Q8_0.estimateVramSavingsMB(32768, 20);
            assertTrue(savings > 0, "Should return positive savings for unknown size");
            // sanity: not larger than an absurd amount
            assertTrue(savings < 100_000, "Savings should be reasonable");
        }
        
        @Test
        @DisplayName("Should not crash for negative context length (current behaviour)")
        void negativeContextShouldNotCrash() {
            // Current implementation would compute negative savings, but shouldn't throw.
            assertDoesNotThrow(() -> KVCacheType.Q4_0.estimateVramSavingsMB(-100, 8));
        }
    }
}