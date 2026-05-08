package ffm.llama.session;

import ffm.llama.batch.BatchDecoder;
import ffm.llama.cache.KvCacheManager;
import ffm.llama.config.ModelConfig;
import ffm.llama.batch.BatchFactory;
import ffm.llama.batch.LlamaBatch;
import ffm.llama.context.LlamaContext;
import ffm.llama.model.LlamaModel;
import ffm.llama.context.state.CachedContextState;
import ffm.llama.sampling.LlamaSampler;
import ffm.llama.session.strategy.ContextStrategy;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;


import java.util.Optional;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class GenerationSessionTest {
    
    @Mock private LlamaModel model;
    @Mock private LlamaContext context;
    @Mock private LlamaSampler sampler;
    @Mock private ContextStrategy contextStrategy;
    @Mock private SessionConfig config;
    @Mock private StateSerializer stateSerializer;
    @Mock private ModelConfig modelConfig;
    @Mock private BatchFactory batchFactory;
    @Mock private LlamaBatch batch;
    @Mock private KvCacheManager kvCache;
    @Mock private BatchDecoder decoder;

    private GenerationSession.Builder baseBuilder;
    
    @BeforeEach
    void setUp() {
        // Context's helpers return the mocks
        lenient().when(context.decoder()).thenReturn(decoder);
        lenient().when(context.kvCache()).thenReturn(kvCache);
        
        // Use lenient stubbing to avoid UnnecessaryStubbingException in tests that don't use these
        lenient().when(context.getModelConfig()).thenReturn(modelConfig);
        lenient().when(modelConfig.getContextSize()).thenReturn(4096);
        lenient().when(modelConfig.getBatchSize()).thenReturn(512);
        
        // Configure BatchFactory to return our mock batch
        lenient().when(batchFactory.createPrefillBatch(any(), anyInt(), anyBoolean())).thenReturn(batch);
        lenient().when(batchFactory.createDecodeBatch(anyInt(), anyInt())).thenReturn(batch);
        
        baseBuilder = GenerationSession.builder()
                .model(model)
                .context(context)
                .sampler(sampler)
                .contextStrategy(contextStrategy)
                .config(config)
                .stateSerializer(stateSerializer)
                .batchFactory(batchFactory);
    }
    
    // ––– Prefill –––
    
    @Nested
    @DisplayName("prefill")
    class Prefill {
        
        @Test
        @DisplayName("Should process tokens in batches and return count")
        void shouldProcessTokensInBatches() {
            when(decoder.decode(any(LlamaBatch.class))).thenReturn(0);
            GenerationSession session = baseBuilder.build();
            int[] tokens = {1, 2, 3};

            int processed = session.prefill(tokens);
            assertEquals(3, processed);
            assertEquals(3, session.getCachePosition());
        }
        
        @Test
        @DisplayName("Should throw if context overflow would occur")
        void shouldThrowIfOverflow() {
            when(modelConfig.getContextSize()).thenReturn(10);
            GenerationSession session = baseBuilder.build();
            int[] tokens = new int[11];
            assertThrows(IllegalStateException.class, () -> session.prefill(tokens));
        }
    }
    
    // ––– Generate –––
    
    @Nested
    @DisplayName("generate")
    class Generate {
        
        @Test
        @DisplayName("Should call sampler, callback, and decode until EOS")
        void shouldCallSamplerCallbackAndDecodeUntilEOS() {
            when(model.getEosToken()).thenReturn(42);
            when(model.tokenToString(anyInt())).thenReturn("A");
            when(decoder.decode(any(LlamaBatch.class))).thenReturn(0);
            when(sampler.sample(context, -1)).thenReturn(1, 2, 42);  // 2 tokens then EOS
            when(contextStrategy.needsManagement(anyInt(), eq(context))).thenReturn(false);
            
            Consumer<String> callback = mock(Consumer.class);
            
            GenerationSession session = baseBuilder.build();
            int generated = session.generate(callback, 0);
            
            assertEquals(2, generated);
            verify(callback, times(2)).accept("A");
        }
        
        @Test
        @DisplayName("Should invoke context management when needed")
        void shouldInvokeContextManagementWhenNeeded() {
            when(model.getEosToken()).thenReturn(42);
            when(model.tokenToString(anyInt())).thenReturn("X");
            when(decoder.decode(any(LlamaBatch.class))).thenReturn(0);
            when(sampler.sample(context, -1)).thenReturn(1, 42);
            when(contextStrategy.needsManagement(anyInt(), eq(context)))
                    .thenReturn(true, false);
            when(contextStrategy.manage(anyInt(), any(), eq(context)))
                    .thenReturn(ContextStrategy.ManagementAction.none());

            GenerationSession session = baseBuilder.build();
            session.generate(t -> {}, 0);
            verify(contextStrategy, atLeastOnce()).manage(anyInt(), any(), eq(context));
        }
    }
    
    // ––– Snapshot –––
    
    @Nested
    @DisplayName("snapshot")
    class Snapshot {
        
        @Test
        @DisplayName("Should delegate to stateSerializer and return result")
        void shouldDelegateToStateSerializer() {
            CachedContextState dummyState = mock(CachedContextState.class);
            when(stateSerializer.snapshot(any(LlamaModel.class), any(LlamaContext.class), any(ModelConfig.class))).thenReturn(Optional.of(dummyState));

            GenerationSession session = baseBuilder.build();
            CachedContextState result = session.snapshot();
            assertSame(dummyState, result);
        }
    }
    
    // ––– Restore –––
    
    @Nested
    @DisplayName("restore")
    class Restore {
        
        @Test
        @DisplayName("Should update position when successful")
        void shouldUpdatePositionWhenSuccessful() {
            CachedContextState state = mock(CachedContextState.class);
            when(state.getNTokens()).thenReturn(100);
            when(stateSerializer.restoreContext(context, state)).thenReturn(true);

            GenerationSession session = baseBuilder.build();
            assertTrue(session.restore(state));
            assertEquals(100, session.getCachePosition());
        }
        
        @Test
        @DisplayName("Should not update position when unsuccessful")
        void shouldNotUpdatePositionWhenUnsuccessful() {
            CachedContextState state = mock(CachedContextState.class);
            when(stateSerializer.restoreContext(context, state)).thenReturn(false);

            GenerationSession session = baseBuilder.build();
            assertFalse(session.restore(state));
            assertEquals(0, session.getCachePosition());
        }
    }
    
    // ––– Reset –––
    
    @Nested
    @DisplayName("reset")
    class Reset {
        
        @Test
        @DisplayName("Should clear KV cache and reset position")
        void shouldClearCacheAndPosition() {
            when(decoder.decode(any(LlamaBatch.class))).thenReturn(0);
            
            GenerationSession session = baseBuilder.build();
            session.prefill(new int[]{1, 2});
            session.reset();
            
            assertEquals(0, session.getCachePosition());
            verify(kvCache).clearKvCache();
        }
    }
    
    // ––– Close –––
    
    @Nested
    @DisplayName("close")
    class Close {
        
        @Test
        @DisplayName("Should close sampler and mark session closed")
        void shouldCloseSamplerAndMarkClosed() {
            GenerationSession session = baseBuilder.build();
            session.close();
            verify(sampler).close();
            assertThrows(IllegalStateException.class, () -> session.prefill(new int[]{1}));
        }
    }
}