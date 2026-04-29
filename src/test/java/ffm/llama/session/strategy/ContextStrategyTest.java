package ffm.llama.session.strategy;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class ContextStrategyTest {
    
    // -----------------------------------------------------------------------
    // EagerClearStrategy
    // -----------------------------------------------------------------------
    @Nested
    @DisplayName("EagerClearStrategy")
    class EagerClear {
        
        private final int contextSize = 4096;
        private final int safetyMargin = 100;
        
        @Test
        @DisplayName("Should require management when position reaches threshold")
        void shouldRequireManagementAtThreshold() {
            EagerClearStrategy strategy = new EagerClearStrategy(contextSize, safetyMargin);
            assertTrue(strategy.needsManagement(contextSize - safetyMargin, null));
        }
        
        @Test
        @DisplayName("Should not require management below threshold")
        void shouldNotRequireManagementBelowThreshold() {
            EagerClearStrategy strategy = new EagerClearStrategy(contextSize, safetyMargin);
            assertFalse(strategy.needsManagement(contextSize - safetyMargin - 1, null));
        }
        
        @Test
        @DisplayName("Management action should always be CLEAR_CACHE")
        void shouldReturnClearCacheAction() {
            EagerClearStrategy strategy = new EagerClearStrategy(contextSize, safetyMargin);
            ContextStrategy.ManagementAction action = strategy.manage(100, new int[0], null);
            assertEquals(ContextStrategy.ManagementAction.clearCache(), action);
        }
    }
    
    // -----------------------------------------------------------------------
    // NoManagementStrategy
    // -----------------------------------------------------------------------
    @Nested
    @DisplayName("NoManagementStrategy")
    class NoManagement {
        
        @Test
        @DisplayName("Should never require management")
        void shouldNeverRequireManagement() {
            NoManagementStrategy strategy = new NoManagementStrategy();
            assertFalse(strategy.needsManagement(0, null));
            assertFalse(strategy.needsManagement(10_000, null));
        }
        
        @Test
        @DisplayName("Management action should always be NONE")
        void shouldReturnNoneAction() {
            NoManagementStrategy strategy = new NoManagementStrategy();
            assertEquals(ContextStrategy.ManagementAction.none(), strategy.manage(0, null, null));
        }
    }
    
    // -----------------------------------------------------------------------
    // SlidingWindowStrategy
    // -----------------------------------------------------------------------
    @Nested
    @DisplayName("SlidingWindowStrategy")
    class SlidingWindow {
        
        private final int contextSize = 2048;
        private final double keepRatio = 0.5;
        private final int safetyMargin = 100;
        
        @Test
        @DisplayName("Should throw on invalid keepRatio")
        void shouldThrowOnInvalidKeepRatio() {
            assertThrows(IllegalArgumentException.class,
                    () -> new SlidingWindowStrategy(2048, 0.0, 0));
            assertThrows(IllegalArgumentException.class,
                    () -> new SlidingWindowStrategy(2048, 1.1, 0));
        }
        
        @Test
        @DisplayName("Should throw on negative safetyMargin")
        void shouldThrowOnNegativeSafetyMargin() {
            assertThrows(IllegalArgumentException.class,
                    () -> new SlidingWindowStrategy(2048, 0.5, -1));
        }
        
        @Test
        @DisplayName("Should require management when position reaches threshold")
        void shouldRequireManagementAtThreshold() {
            SlidingWindowStrategy strategy = new SlidingWindowStrategy(contextSize, keepRatio, safetyMargin);
            assertTrue(strategy.needsManagement(contextSize - safetyMargin, null));
        }
        
        @Test
        @DisplayName("Should not require management below threshold")
        void shouldNotRequireManagementBelowThreshold() {
            SlidingWindowStrategy strategy = new SlidingWindowStrategy(contextSize, keepRatio, safetyMargin);
            assertFalse(strategy.needsManagement(contextSize - safetyMargin - 1, null));
        }
        
        @Test
        @DisplayName("Should return shift-left with computed keep count")
        void shouldReturnShiftLeftWithCorrectKeep() {
            SlidingWindowStrategy strategy = new SlidingWindowStrategy(contextSize, keepRatio, safetyMargin);
            int expectedKeep = (int) (contextSize * keepRatio);
            ContextStrategy.ManagementAction action = strategy.manage(1000, new int[0], null);
            assertEquals(ContextStrategy.ManagementAction.shiftLeft(expectedKeep), action);
        }
        
        @Test
        @DisplayName("Should enforce minimum of 100 tokens if computed keep is less than 100")
        void shouldEnforceMinimumKeepTokens() {
            // 1000*0.05 = 50, so minimum should be 100 (or currentPosition if smaller)
            SlidingWindowStrategy strategy = new SlidingWindowStrategy(1000, 0.05, 0);
            ContextStrategy.ManagementAction action = strategy.manage(80, new int[0], null);
            // currentPosition is 80 < 100, so keep = 80
            assertEquals(ContextStrategy.ManagementAction.shiftLeft(80), action);
        }
        
        @Test
        @DisplayName("Should use 100 as minimum when current position is larger")
        void shouldUseMinimum100WhenPositionLarger() {
            SlidingWindowStrategy strategy = new SlidingWindowStrategy(1000, 0.05, 0);
            ContextStrategy.ManagementAction action = strategy.manage(200, new int[0], null);
            assertEquals(ContextStrategy.ManagementAction.shiftLeft(100), action);
        }
    }
}
