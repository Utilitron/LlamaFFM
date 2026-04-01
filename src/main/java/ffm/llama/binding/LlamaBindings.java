package ffm.llama.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.Charset;

/**
 * Complete FFM bindings for llama.cpp with support for:
 * - Model loading and context creation
 * - Tokenization and detokenization
 * - Batch inference (prefill/decode phases)
 * - Sampling and logit processing
 * - Memory and performance telemetry
 * - Offloading configuration
 */
public class LlamaBindings {

    private static final Linker linker = Linker.nativeLinker();
    private static final SymbolLookup lookup = SymbolLookup.loaderLookup();

    // Static field to prevent GC of the native callback stub
    private static MemorySegment LOG_STUB;
    private static volatile boolean loggingEnabled = false;

    static {
        // Look for a specific environment variable, fallback to a default name
        String llamaPath = System.getenv("LLAMA_LIB_PATH");

        if (llamaPath != null) {
            System.load(llamaPath);
        } else {
            // Fallback: try to load from the system's standard library paths (LD_LIBRARY_PATH)
            System.loadLibrary("llama");
        }

        installLogCallback();
    }

    // ============================================================================
    // STRUCT LAYOUTS - Critical for proper FFM memory management
    // ============================================================================

    /**
     * Layout for llama_model_params
     * Controls model loading behavior and initial offloading strategy
     */
    public static final StructLayout MODEL_PARAMS_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("devices"),                        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
            ValueLayout.ADDRESS.withName("tensor_buft_overrides"),          // Writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
            ValueLayout.JAVA_INT.withName("n_gpu_layers"),                  // Number of layers to offload to GPU (Tier 0)
            ValueLayout.JAVA_INT.withName("split_mode"),                    // 0=NONE, 1=LAYER, 2=ROW
            ValueLayout.JAVA_INT.withName("main_gpu"),                      // Main GPU to use
            MemoryLayout.paddingLayout(4),                          // 4 bytes (Alignment)
            ValueLayout.ADDRESS.withName("tensor_split"),                   // Per-GPU tensor distribution
            ValueLayout.ADDRESS.withName("progress_callback"),              // Progress callback function pointer
            ValueLayout.ADDRESS.withName("progress_callback_user_data"),    // User data for callback
            ValueLayout.ADDRESS.withName("kv_overrides"),                   // Pointer to vector containing overrides
            ValueLayout.JAVA_BYTE.withName("vocab_only"),                   // Only load vocabulary
            ValueLayout.JAVA_BYTE.withName("use_mmap"),                     // Use mmap for model loading
            ValueLayout.JAVA_BYTE.withName("use_direct_io"),                // Use direct io, takes precedence over use_mmap when supported
            ValueLayout.JAVA_BYTE.withName("use_mlock"),                    // Lock model in RAM (prevents swap to disk)
            ValueLayout.JAVA_BYTE.withName("check_tensors"),                // Validate tensor data
            ValueLayout.JAVA_BYTE.withName("use_extra_bufts"),              // Use extra buffer types (used for weight repacking)
            ValueLayout.JAVA_BYTE.withName("no_host"),                      // Bypass host buffer allowing extra buffers to be used
            ValueLayout.JAVA_BYTE.withName("no_alloc")                      // Only load metadata and simulate memory allocations
    ).withName("llama_model_params");

    /**
     * Layout for llama_context_params
     * Controls inference behavior, KV cache, and memory allocation
     */
    public static final StructLayout CONTEXT_PARAMS_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("n_ctx"),                         // Context size (max tokens)
            ValueLayout.JAVA_INT.withName("n_batch"),                       // Batch size for prompt processing
            ValueLayout.JAVA_INT.withName("n_ubatch"),                      // Physical batch size
            ValueLayout.JAVA_INT.withName("n_seq_max"),                     // Max sequences (for parallel decoding)
            ValueLayout.JAVA_INT.withName("n_threads"),                     // Number of CPU threads
            ValueLayout.JAVA_INT.withName("n_threads_batch"),               // Threads for batch processing
            ValueLayout.JAVA_INT.withName("rope_scaling_type"),             // RoPE scaling type
            ValueLayout.JAVA_INT.withName("pooling_type"),                  // Pooling type for embeddings
            ValueLayout.JAVA_INT.withName("attention_type"),                // Attention implementation
            ValueLayout.JAVA_INT.withName("flash_attn_type"),               // When to enable Flash Attention
            ValueLayout.JAVA_FLOAT.withName("rope_freq_base"),              // RoPE frequency base
            ValueLayout.JAVA_FLOAT.withName("rope_freq_scale"),             // RoPE frequency scale
            ValueLayout.JAVA_FLOAT.withName("yarn_ext_factor"),             // YaRN extrapolation factor
            ValueLayout.JAVA_FLOAT.withName("yarn_attn_factor"),            // YaRN attention factor
            ValueLayout.JAVA_FLOAT.withName("yarn_beta_fast"),              // YaRN beta fast
            ValueLayout.JAVA_FLOAT.withName("yarn_beta_slow"),              // YaRN beta slow
            ValueLayout.JAVA_INT.withName("yarn_orig_ctx"),                 // YaRN original context
            ValueLayout.JAVA_FLOAT.withName("defrag_thold"),                // KV cache defragmentation threshold
            ValueLayout.ADDRESS.withName("cb_eval"),                        // Eval callback
            ValueLayout.ADDRESS.withName("cb_eval_user_data"),              // User data for eval callback
            ValueLayout.JAVA_INT.withName("type_k"),                        // KV cache data type for K
            ValueLayout.JAVA_INT.withName("type_v"),                        // KV cache data type for V
            ValueLayout.ADDRESS.withName("abort_callback"),                 // Abort callback: returns true, execution of llama_decode() will be aborted
            ValueLayout.ADDRESS.withName("abort_callback_data"),            // Currently works only with CPU execution
            ValueLayout.JAVA_BYTE.withName("embeddings"),                   // Generate embeddings only
            ValueLayout.JAVA_BYTE.withName("offload_kqv"),                  // Offload KQV to GPU
            ValueLayout.JAVA_BYTE.withName("no_perf"),                      // Measure performance timings
            ValueLayout.JAVA_BYTE.withName("op_offload"),                   // Offload host tensor operations to device
            ValueLayout.JAVA_BYTE.withName("swa_full"),                     // Use full-size SWA cache
            ValueLayout.JAVA_BYTE.withName("kv_unified"),                   // Use a unified buffer across the input sequences when computing the attention
            MemoryLayout.paddingLayout(2),                          // 4 bytes (Alignment)
            ValueLayout.ADDRESS.withName("samplers"),                       // [EXPERIMENTAL]
            ValueLayout.JAVA_LONG.withName("n_samplers")                    // Backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
    ).withByteAlignment(8).withName("llama_context_params");

    /**
     * Layout for llama_batch
     * Used for efficient batch processing during prefill and decode
     */
    public static final StructLayout BATCH_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("n_tokens"),                      // Number of tokens in batch
            MemoryLayout.paddingLayout(4),                          // 4 bytes (Alignment)
            ValueLayout.ADDRESS.withName("token"),                          // Token IDs (int32_t*)
            ValueLayout.ADDRESS.withName("embd"),                           // Embeddings (float*)
            ValueLayout.ADDRESS.withName("pos"),                            // Positions (llama_pos*)
            ValueLayout.ADDRESS.withName("n_seq_id"),                       // Number of sequence IDs per token
            ValueLayout.ADDRESS.withName("seq_id"),                         // Sequence IDs (llama_seq_id**)
            ValueLayout.ADDRESS.withName("logits")                          // Whether to compute logits (int8_t*)
    ).withName("llama_batch");

    /**
     * Layout for llama_sampler_chain_params
     */
    public static final StructLayout SAMPLER_CHAIN_PARAMS_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_BYTE.withName("no_perf")                       // Whether to measure performance
    ).withName("llama_sampler_chain_params");

    // ============================================================================
    // CORE LIFECYCLE FUNCTIONS
    // ============================================================================

    public static final MethodHandle llama_backend_init;
    public static final MethodHandle llama_backend_free;
    public static final MethodHandle llama_model_default_params;
    public static final MethodHandle llama_context_default_params;
    public static final MethodHandle llama_sampler_chain_default_params;
    public static final MethodHandle llama_model_load_from_file;
    public static final MethodHandle llama_model_free;
    public static final MethodHandle llama_init_from_model;
    public static final MethodHandle llama_free;

    // ============================================================================
    // VOCAB
    // ============================================================================

    public static final MethodHandle llama_model_get_vocab;
    public static final MethodHandle llama_vocab_n_tokens;
    public static final MethodHandle llama_vocab_bos;
    public static final MethodHandle llama_vocab_eos;
    public static final MethodHandle llama_vocab_eot;
    public static final MethodHandle llama_vocab_nl;

    // ============================================================================
    // MODEL INTROSPECTION
    // ============================================================================

    public static final MethodHandle llama_model_n_ctx_train;
    public static final MethodHandle llama_model_n_embd;
    public static final MethodHandle llama_model_n_layer;
    public static final MethodHandle llama_model_desc;
    public static final MethodHandle llama_model_size;
    public static final MethodHandle llama_model_n_params;

    // ============================================================================
    // TOKENIZATION - Text processing
    // ============================================================================

    public static final MethodHandle llama_tokenize;
    public static final MethodHandle llama_token_to_piece;
    public static final MethodHandle llama_detokenize;

    // ============================================================================
    // BATCH PROCESSING - For efficient prefill/decode phases
    // ============================================================================

    public static final MethodHandle llama_batch_get_one;
    public static final MethodHandle llama_batch_init;
    public static final MethodHandle llama_batch_free;
    public static final MethodHandle llama_decode;
    public static final MethodHandle llama_get_logits;
    public static final MethodHandle llama_get_logits_ith;
    public static final MethodHandle llama_get_embeddings;
    public static final MethodHandle llama_get_embeddings_ith;
    public static final MethodHandle llama_get_embeddings_seq;

    // ============================================================================
    // SAMPLING - Token generation
    // ============================================================================

    public static final MethodHandle llama_sampler_chain_init;
    public static final MethodHandle llama_sampler_chain_add;
    public static final MethodHandle llama_sampler_init_greedy;
    public static final MethodHandle llama_sampler_init_dist;
    public static final MethodHandle llama_sampler_init_top_k;
    public static final MethodHandle llama_sampler_init_top_p;
    public static final MethodHandle llama_sampler_init_min_p;
    public static final MethodHandle llama_sampler_init_temp;
    public static final MethodHandle llama_sampler_sample;
    public static final MethodHandle llama_sampler_free;

    // ============================================================================
    // KV CACHE MANAGEMENT
    // ============================================================================

    public static final MethodHandle llama_get_memory;
    public static final MethodHandle llama_memory_clear;
    public static final MethodHandle llama_memory_seq_rm;
    public static final MethodHandle llama_memory_seq_cp;
    public static final MethodHandle llama_memory_seq_keep;
    public static final MethodHandle llama_memory_seq_add;
    public static final MethodHandle llama_memory_seq_div;
    public static final MethodHandle llama_memory_seq_pos_max;

    // ============================================================================
    // STATE MANAGEMENT - For persistence and offloading
    // ============================================================================

    public static final MethodHandle llama_state_get_size;
    public static final MethodHandle llama_state_get_data;
    public static final MethodHandle llama_state_set_data;
    public static final MethodHandle llama_state_seq_get_size;
    public static final MethodHandle llama_state_seq_get_data;
    public static final MethodHandle llama_state_seq_set_data;
    public static final MethodHandle llama_state_seq_save_file;
    public static final MethodHandle llama_state_seq_load_file;

    // ============================================================================
    // PERFORMANCE TELEMETRY - For monitoring offloading efficiency
    // ============================================================================

    public static final MethodHandle llama_perf_context;
    public static final MethodHandle llama_perf_context_print;
    public static final MethodHandle llama_perf_context_reset;
    public static final MethodHandle llama_perf_sampler_print;
    public static final MethodHandle llama_perf_sampler_reset;

    // ============================================================================
    // INITIALIZATION - Bind all native functions
    // ============================================================================

    static {
        try {
            // Core Lifecycle
            llama_backend_init = linker.downcallHandle(
                    lookup.find("llama_backend_init").orElseThrow(),
                    FunctionDescriptor.ofVoid()
            );

            llama_backend_free = linker.downcallHandle(
                    lookup.find("llama_backend_free").orElseThrow(),
                    FunctionDescriptor.ofVoid()
            );

            // Default parameters - return structs by value
            llama_model_default_params = linker.downcallHandle(
                    lookup.find("llama_model_default_params").orElseThrow(),
                    FunctionDescriptor.of(MODEL_PARAMS_LAYOUT)
            );

            llama_context_default_params = linker.downcallHandle(
                    lookup.find("llama_context_default_params").orElseThrow(),
                    FunctionDescriptor.of(CONTEXT_PARAMS_LAYOUT)
            );

            llama_sampler_chain_default_params = linker.downcallHandle(
                    lookup.find("llama_sampler_chain_default_params").orElseThrow(),
                    FunctionDescriptor.of(SAMPLER_CHAIN_PARAMS_LAYOUT)
            );

            llama_model_load_from_file = linker.downcallHandle(
                    lookup.find("llama_model_load_from_file").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, MODEL_PARAMS_LAYOUT)
            );

            llama_model_free = linker.downcallHandle(
                    lookup.find("llama_model_free").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            llama_init_from_model = linker.downcallHandle(
                    lookup.find("llama_init_from_model").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, CONTEXT_PARAMS_LAYOUT)
            );

            llama_free = linker.downcallHandle(
                    lookup.find("llama_free").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            // Vocab
            llama_model_get_vocab = linker.downcallHandle(
                    lookup.find("llama_model_get_vocab").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );

            llama_vocab_n_tokens = linker.downcallHandle(
                    lookup.find("llama_vocab_n_tokens").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

            llama_vocab_bos = linker.downcallHandle(
                    lookup.find("llama_vocab_bos").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

            llama_vocab_eos = linker.downcallHandle(
                    lookup.find("llama_vocab_eos").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

            llama_vocab_eot = linker.downcallHandle(
                    lookup.find("llama_vocab_eot").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

            llama_vocab_nl = linker.downcallHandle(
                    lookup.find("llama_vocab_nl").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

            // Model introspection
            llama_model_n_ctx_train = linker.downcallHandle(
                    lookup.find("llama_model_n_ctx_train").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

            llama_model_n_embd = linker.downcallHandle(
                    lookup.find("llama_model_n_embd").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

            llama_model_n_layer = linker.downcallHandle(
                    lookup.find("llama_model_n_layer").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
            );

            llama_model_desc = linker.downcallHandle(
                    lookup.find("llama_model_desc").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            llama_model_size = linker.downcallHandle(
                    lookup.find("llama_model_size").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
            );

            llama_model_n_params = linker.downcallHandle(
                    lookup.find("llama_model_n_params").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
            );

            // Tokenization
            llama_tokenize = linker.downcallHandle(
                    lookup.find("llama_tokenize").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN)
            );

            llama_token_to_piece = linker.downcallHandle(
                    lookup.find("llama_token_to_piece").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN)
            );

            llama_detokenize = linker.downcallHandle(
                    lookup.find("llama_detokenize").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN)
            );

            // Batch processing
            llama_batch_get_one = linker.downcallHandle(
                    lookup.find("llama_batch_get_one").orElseThrow(),
                    FunctionDescriptor.of(BATCH_LAYOUT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            llama_batch_init = linker.downcallHandle(
                    lookup.find("llama_batch_init").orElseThrow(),
                    FunctionDescriptor.of(BATCH_LAYOUT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
            );

            llama_batch_free = linker.downcallHandle(
                    lookup.find("llama_batch_free").orElseThrow(),
                    FunctionDescriptor.ofVoid(BATCH_LAYOUT)
            );

            llama_decode = linker.downcallHandle(
                    lookup.find("llama_decode").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, BATCH_LAYOUT)
            );

           llama_get_logits = linker.downcallHandle(
                    lookup.find("llama_get_logits").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );

            llama_get_logits_ith = linker.downcallHandle(
                    lookup.find("llama_get_logits_ith").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            llama_get_embeddings = linker.downcallHandle(
                    lookup.find("llama_get_embeddings").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );

            llama_get_embeddings_ith = linker.downcallHandle(
                    lookup.find("llama_get_embeddings_ith").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            llama_get_embeddings_seq = linker.downcallHandle(
                    lookup.find("llama_get_embeddings_seq").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            // Sampling
            llama_sampler_chain_init = linker.downcallHandle(
                    lookup.find("llama_sampler_chain_init").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, SAMPLER_CHAIN_PARAMS_LAYOUT)
            );

            llama_sampler_chain_add = linker.downcallHandle(
                    lookup.find("llama_sampler_chain_add").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );

            llama_sampler_init_greedy = linker.downcallHandle(
                    lookup.find("llama_sampler_init_greedy").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS)
            );

            llama_sampler_init_dist = linker.downcallHandle(
                    lookup.find("llama_sampler_init_dist").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            llama_sampler_init_top_k = linker.downcallHandle(
                    lookup.find("llama_sampler_init_top_k").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            llama_sampler_init_top_p = linker.downcallHandle(
                    lookup.find("llama_sampler_init_top_p").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG)
            );

            llama_sampler_init_min_p = linker.downcallHandle(
                    lookup.find("llama_sampler_init_min_p").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG)
            );

            llama_sampler_init_temp = linker.downcallHandle(
                    lookup.find("llama_sampler_init_temp").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT)
            );

            llama_sampler_sample = linker.downcallHandle(
                    lookup.find("llama_sampler_sample").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            llama_sampler_free = linker.downcallHandle(
                    lookup.find("llama_sampler_free").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            // KV cache management
            llama_get_memory = linker.downcallHandle(
                    lookup.find("llama_get_memory").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );

            llama_memory_clear = linker.downcallHandle(
                    lookup.find("llama_memory_clear").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN)
            );

            llama_memory_seq_rm = linker.downcallHandle(
                    lookup.find("llama_memory_seq_rm").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
            );

            llama_memory_seq_cp = linker.downcallHandle(
                    lookup.find("llama_memory_seq_cp").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
            );

            llama_memory_seq_keep = linker.downcallHandle(
                    lookup.find("llama_memory_seq_keep").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            llama_memory_seq_add = linker.downcallHandle(
                    lookup.find("llama_memory_seq_add").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
            );

            llama_memory_seq_div = linker.downcallHandle(
                    lookup.find("llama_memory_seq_div").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
            );

            llama_memory_seq_pos_max = linker.downcallHandle(
                    lookup.find("llama_memory_seq_pos_max").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            // State management
            llama_state_get_size = linker.downcallHandle(
                    lookup.find("llama_state_get_size").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
            );

            llama_state_get_data = linker.downcallHandle(
                    lookup.find("llama_state_get_data").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            llama_state_set_data = linker.downcallHandle(
                    lookup.find("llama_state_set_data").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            llama_state_seq_get_size = linker.downcallHandle(
                    lookup.find("llama_state_seq_get_size").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
            );

            llama_state_seq_get_data = linker.downcallHandle(
                    lookup.find("llama_state_seq_get_data").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT)
            );

            llama_state_seq_set_data = linker.downcallHandle(
                    lookup.find("llama_state_seq_set_data").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT)
            );

            llama_state_seq_save_file = linker.downcallHandle(
                    lookup.find("llama_state_seq_save_file").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            llama_state_seq_load_file = linker.downcallHandle(
                    lookup.find("llama_state_seq_load_file").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            // Performance telemetry
            llama_perf_context = linker.downcallHandle(
                    lookup.find("llama_perf_context").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );

            llama_perf_context_print = linker.downcallHandle(
                    lookup.find("llama_perf_context_print").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            llama_perf_context_reset = linker.downcallHandle(
                    lookup.find("llama_perf_context_reset").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            llama_perf_sampler_print = linker.downcallHandle(
                    lookup.find("llama_perf_sampler_print").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            llama_perf_sampler_reset = linker.downcallHandle(
                    lookup.find("llama_perf_sampler_reset").orElseThrow(),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize LlamaBindings", e);
        }
    }

    /**
     * Initialize the llama.cpp backend
     * Must be called once before using any other functions
     */
    public static void init() {
        try {
            llama_backend_init.invoke();
        } catch (Throwable t) {
            throw new RuntimeException("Failed to initialize llama backend", t);
        }
    }

    /**
     * Free the llama.cpp backend
     * Should be called at shutdown
     */
    public static void free() {
        try {
            llama_backend_free.invoke();
        } catch (Throwable t) {
            throw new RuntimeException("Failed to free llama backend", t);
        }
    }

    public static void enableLogging() {
        loggingEnabled = true;
    }

    public static void disableLogging() {
        loggingEnabled = false;
    }

    private static void installLogCallback() {
        try {
            // Use loaderLookup – the library is already loaded
            SymbolLookup loaderLookup = SymbolLookup.loaderLookup();
            MemorySegment logSetAddr = loaderLookup.find("llama_log_set")
                    .orElseThrow(() -> new RuntimeException("llama_log_set not found"));

            MethodHandle logSet = linker.downcallHandle(
                    logSetAddr,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
            );

            MethodHandle logger = MethodHandles.lookup().findStatic(
                    LlamaBindings.class,
                    "logCallback",
                    MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class)
            );

            LOG_STUB = linker.upcallStub(
                    logger,
                    FunctionDescriptor.ofVoid(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS
                    ),
                    Arena.global()
            );

            // Install the callback (user_data = NULL)
            logSet.invokeExact(LOG_STUB, MemorySegment.NULL);

        } catch (Throwable t) {
            System.err.println("[WARN] Failed to install llama logger: " + t);
            LOG_STUB = null;
        }
    }

    private static void logCallback(int level, MemorySegment text, MemorySegment user_data) {
        if (!loggingEnabled) return;  // muted

        String msg = text == null ? "" : text.getString(0, Charset.defaultCharset());
        System.err.printf("[llama level %d] %s%n", level, msg);
    }
}