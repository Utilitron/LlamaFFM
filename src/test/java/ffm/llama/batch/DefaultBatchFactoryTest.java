package ffm.llama.batch;

import org.junit.jupiter.api.Test;
import org.mockito.InOrder;
import org.mockito.MockedConstruction;

import static org.mockito.Mockito.*;

public class DefaultBatchFactoryTest {
    
    @Test
    void shouldSetEmitLogitsOnlyOnLastTokenWhenOutputLogitsTrue() {
        try (MockedConstruction<LlamaBatch> mocked = mockConstruction(LlamaBatch.class)) {
            DefaultBatchFactory factory = new DefaultBatchFactory();
            int[] tokens = {10, 20, 30};
            LlamaBatch batch = factory.createPrefillBatch(tokens, 5, true);
            
            LlamaBatch mockBatch = mocked.constructed().get(0);
            InOrder inOrder = inOrder(mockBatch);
            inOrder.verify(mockBatch).add(10, 5, 0, false);
            inOrder.verify(mockBatch).add(20, 6, 0, false);
            inOrder.verify(mockBatch).add(30, 7, 0, true);
            verifyNoMoreInteractions(mockBatch);
        }
    }
    
    @Test
    void shouldSetEmitLogitsFalseForAllTokensWhenOutputLogitsFalse() {
        try (MockedConstruction<LlamaBatch> mocked = mockConstruction(LlamaBatch.class)) {
            DefaultBatchFactory factory = new DefaultBatchFactory();
            int[] tokens = {10, 20};
            factory.createPrefillBatch(tokens, 0, false);
            
            LlamaBatch mockBatch = mocked.constructed().get(0);
            verify(mockBatch, times(2)).add(anyInt(), anyInt(), eq(0), eq(false));
            // Optionally verify positions etc.
        }
    }
    
}
