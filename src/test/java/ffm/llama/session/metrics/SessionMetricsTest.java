package ffm.llama.session.metrics;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class SessionMetricsTest {
    
    private SessionMetrics metrics;
    
    @BeforeEach
    void setUp() {
        metrics = new SessionMetrics();
    }
    
    @Nested
    @DisplayName("Initial state")
    class InitialState {
        
        @Test
        @DisplayName("Should have zero prefill tokens and nanos")
        void shouldHaveZeroPrefill() {
            assertEquals(0, metrics.getPrefillTokens());
            assertEquals(0.0, metrics.getPrefillTokensPerSecond());
        }
        
        @Test
        @DisplayName("Should have zero generation tokens and nanos")
        void shouldHaveZeroGeneration() {
            assertEquals(0, metrics.getGenerationTokens());
            assertEquals(0.0, metrics.getGenerationTokensPerSecond());
        }
        
        @Test
        @DisplayName("Should have zero total tokens and throughput")
        void shouldHaveZeroTotal() {
            assertEquals(0, metrics.getTotalTokens());
            assertEquals(0.0, metrics.getTotalTokensPerSecond());
        }
        
        @Test
        @DisplayName("Should have zero context events")
        void shouldHaveZeroEvents() {
            assertEquals(0, metrics.getContextShifts());
            assertEquals(0, metrics.getContextClears());
        }
    }
    
    @Nested
    @DisplayName("Prefill recording")
    class Prefill {
        
        @Test
        @DisplayName("Should accumulate prefill tokens and nanos")
        void shouldAccumulatePrefill() {
            metrics.recordPrefill(100, 1_000_000_000L); // 1 second
            assertEquals(100, metrics.getPrefillTokens());
            // 100 tokens per second
            assertEquals(100.0, metrics.getPrefillTokensPerSecond(), 0.01);
            // 1 second latency
            assertEquals(1000.0, metrics.getPrefillLatencyMs(), 0.01);
        }
        
        @Test
        @DisplayName("Should sum multiple prefill records")
        void shouldSumMultiplePrefills() {
            metrics.recordPrefill(100, 2_000_000_000L);
            metrics.recordPrefill(50, 1_000_000_000L);
            assertEquals(150, metrics.getPrefillTokens());
            assertEquals(150.0 * 1e9 / 3_000_000_000L, metrics.getPrefillTokensPerSecond(), 0.01);
        }
        
        @Test
        @DisplayName("Should return zero throughput with zero nanos")
        void shouldReturnZeroThroughputWhenNanosZero() {
            metrics.recordPrefill(100, 0);
            assertEquals(0.0, metrics.getPrefillTokensPerSecond());
        }
    }
    
    @Nested
    @DisplayName("Generation recording")
    class Generation {
        
        @Test
        @DisplayName("Should accumulate generation tokens and nanos")
        void shouldAccumulateGeneration() {
            metrics.recordGeneration(200, 2_000_000_000L); // 2 seconds
            assertEquals(200, metrics.getGenerationTokens());
            assertEquals(100.0, metrics.getGenerationTokensPerSecond(), 0.01);
            assertEquals(2000.0, metrics.getGenerationLatencyMs(), 0.01);
        }
        
        @Test
        @DisplayName("Should sum multiple generation records")
        void shouldSumMultipleGenerations() {
            metrics.recordGeneration(200, 4_000_000_000L);
            metrics.recordGeneration(100, 1_000_000_000L);
            assertEquals(300, metrics.getGenerationTokens());
            assertEquals(300.0 * 1e9 / 5_000_000_000L, metrics.getGenerationTokensPerSecond(), 0.01);
        }
    }
    
    @Nested
    @DisplayName("Combined metrics")
    class Combined {
        
        @Test
        @DisplayName("Should sum prefill and generation into total tokens")
        void shouldSumTotalTokens() {
            metrics.recordPrefill(100, 1_000_000_000L);
            metrics.recordGeneration(200, 2_000_000_000L);
            assertEquals(300, metrics.getTotalTokens());
        }
        
        @Test
        @DisplayName("Should compute overall throughput from combined nanos")
        void shouldComputeOverallThroughput() {
            metrics.recordPrefill(100, 1_000_000_000L);
            metrics.recordGeneration(200, 3_000_000_000L);
            double expected = 300.0 * 1e9 / 4_000_000_000L;
            assertEquals(expected, metrics.getTotalTokensPerSecond(), 0.01);
        }
        
        @Test
        @DisplayName("Should compute total latency as sum of prefill and generation")
        void shouldComputeTotalLatency() {
            metrics.recordPrefill(100, 500_000_000L);   // 500 ms
            metrics.recordGeneration(200, 1_500_000_000L); // 1500 ms
            assertEquals(2000.0, metrics.getTotalLatencyMs(), 0.01);
        }
    }
    
    @Nested
    @DisplayName("Context events")
    class ContextEvents {
        
        @Test
        @DisplayName("Should count context shifts")
        void shouldCountShifts() {
            metrics.recordContextShift();
            metrics.recordContextShift();
            assertEquals(2, metrics.getContextShifts());
        }
        
        @Test
        @DisplayName("Should count context clears")
        void shouldCountClears() {
            metrics.recordContextClear();
            assertEquals(1, metrics.getContextClears());
        }
    }
    
    @Test
    @DisplayName("Should reset all counters to zero")
    void shouldResetAllCounters() {
        metrics.recordPrefill(100, 1_000_000_000L);
        metrics.recordGeneration(200, 2_000_000_000L);
        metrics.recordContextShift();
        metrics.recordContextClear();
        
        metrics.reset();
        
        assertEquals(0, metrics.getPrefillTokens());
        assertEquals(0, metrics.getGenerationTokens());
        assertEquals(0, metrics.getTotalTokens());
        assertEquals(0.0, metrics.getPrefillTokensPerSecond());
        assertEquals(0.0, metrics.getGenerationTokensPerSecond());
        assertEquals(0.0, metrics.getTotalTokensPerSecond());
        assertEquals(0, metrics.getContextShifts());
        assertEquals(0, metrics.getContextClears());
    }
    
    @Test
    @DisplayName("Summary should contain key metrics")
    void summaryShouldContainMetrics() {
        metrics.recordPrefill(100, 1_000_000_000L);
        metrics.recordGeneration(200, 2_000_000_000L);
        String s = metrics.summary();
        assertTrue(s.contains("prefill=100"));
        assertTrue(s.contains("gen=200"));
        assertTrue(s.contains("total=300"));
        assertTrue(s.contains("shifts=0"));
        assertTrue(s.contains("clears=0"));
    }
    
    @Test
    @DisplayName("toString should return same as summary")
    void toStringShouldEqualSummary() {
        metrics.recordPrefill(100, 1_000_000_000L);
        assertEquals(metrics.summary(), metrics.toString());
    }
}
