package ffm.llama.batch;

import ffm.llama.IntegrationTestBase;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class LlamaBatchIT extends IntegrationTestBase {
    
    private LlamaBatch batch;
    
    @AfterEach
    void tearDown() {
        if (batch != null) {
            // close is idempotent in llama.cpp? Not guaranteed, but we try to clean up.
            try { batch.close(); } catch (Exception ignored) {}
            batch = null;
        }
    }
    
    @Test
    @DisplayName("Should create empty batch with correct capacity")
    void shouldCreateEmptyBatch() {
        batch = LlamaBatch.create(5);
        assertEquals(0, batch.size());
        assertEquals(5, batch.capacity());
        assertTrue(batch.isEmpty());
        assertFalse(batch.isFull());
    }
    
    @Test
    @DisplayName("Should add tokens up to capacity and detect full")
    void shouldAddTokensAndBecomeFull() {
        batch = LlamaBatch.create(3);
        batch.add(10, 0, 0, false);
        batch.add(20, 1, 0, false);
        batch.add(30, 2, 0, true);
        assertEquals(3, batch.size());
        assertTrue(batch.isFull());
        assertFalse(batch.isEmpty());
    }
    
    @Test
    @DisplayName("Should throw when adding beyond capacity")
    void shouldThrowWhenAddingToFullBatch() {
        batch = LlamaBatch.create(2);
        batch.add(1, 0, 0, false);
        batch.add(2, 1, 0, true);
        assertThrows(IllegalStateException.class,
                () -> batch.add(3, 2, 0, false));
    }
    
    @Test
    @DisplayName("Should clear batch and reset size")
    void shouldClearBatch() {
        batch = LlamaBatch.create(4);
        batch.add(1, 0, 0, false);
        batch.add(2, 1, 0, true);
        assertEquals(2, batch.size());
        batch.clear();
        assertEquals(0, batch.size());
        assertTrue(batch.isEmpty());
        assertFalse(batch.isFull());
        // after clear, adding again should work
        batch.add(3, 0, 0, false);
        assertEquals(1, batch.size());
    }
    
    @Test
    @DisplayName("Should create single-token batch with forSingleToken")
    void shouldCreateForSingleToken() {
        batch = LlamaBatch.forSingleToken(42, 7, 0);
        assertEquals(1, batch.size());
        assertTrue(batch.isFull()); // capacity = 1
    }
    
    @Test
    @DisplayName("Should create prefill batch with forTokens and control logits")
    void shouldCreateForTokens() {
        int[] tokens = {100, 200, 300};
        batch = LlamaBatch.forTokens(tokens, 10, 0, true);
        assertEquals(3, batch.size());
    }
    
    @Test
    @DisplayName("Should free native memory on close without exception")
    void shouldCloseWithoutException() {
        batch = LlamaBatch.create(1);
        assertDoesNotThrow(() -> batch.close());
        batch = null; // avoid double close in tearDown
    }
    
    @Test
    @DisplayName("Should handle batch with zero capacity? (edge case)")
    void shouldRejectZeroCapacity() {
        assertThrows(RuntimeException.class, () -> new LlamaBatch(0, 1));
    }
}
