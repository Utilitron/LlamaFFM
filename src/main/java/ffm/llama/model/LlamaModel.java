package ffm.llama.model;

import ffm.llama.binding.LlamaBindings;
import ffm.llama.config.ModelConfig;
import ffm.llama.utils.TemplateDetector;

import java.lang.foreign.*;
import java.lang.invoke.VarHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Represents a loaded llama.cpp model configuration.
 * Manages the model lifecycle and provides introspection capabilities.
 */
public class LlamaModel implements AutoCloseable {

    private final Arena modelArena;
    private final MemorySegment modelPtr;
    private final MemorySegment vocabPtr;
    private final String modelPath;
    private final ModelConfig modelConfig;

    // Model metadata (cached after loading)
    private final int nVocab;
    private final int nCtxTrain;
    private final int nEmbd;
    private final int nLayer;
    private final long nParams;
    private final long modelSizeBytes;

    /**
     * Load a model from disk
     */
    public LlamaModel(String path) {
        this(path, null);
    }

    /**
     * Load a model with explicit model configuration
     * 
     * @param path Path to .gguf model file
     * @param modelConfig Model configuration (null = auto-detect)
     */
    public LlamaModel(String path, ModelConfig modelConfig) {
        this.modelPath = path;
        this.modelArena = Arena.ofShared();

        try {
            // Allocate path string in model's arena
            MemorySegment cPath = modelArena.allocateFrom(path);

            // Create model parameters struct
            MemorySegment modelParams = modelArena.allocate(LlamaBindings.MODEL_PARAMS_LAYOUT);

            // Get default parameters
            MemorySegment defaultParams = (MemorySegment) LlamaBindings.llama_model_default_params.invoke(modelArena);
            MemorySegment.copy(defaultParams, 0, modelParams, 0, LlamaBindings.MODEL_PARAMS_LAYOUT.byteSize());

            // If model config provided, override specific fields
            if (modelConfig != null) {
                this.modelConfig = modelConfig;
                VarHandle gpuLayersHandle = LlamaBindings.MODEL_PARAMS_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("n_gpu_layers"));
                gpuLayersHandle.set(modelParams, 0L, modelConfig.getGpuLayers());
                VarHandle mmapHandle = LlamaBindings.MODEL_PARAMS_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("use_mmap"));
                mmapHandle.set(modelParams, 0L, (byte) (modelConfig.isUseMmap() ? 1 : 0));
                VarHandle mlockHandle = LlamaBindings.MODEL_PARAMS_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("use_mlock"));
                mlockHandle.set(modelParams, 0L, (byte) (modelConfig.isUseMlock() ? 1 : 0));
            } else {
                // Auto-detect will happen after we know model size
                this.modelConfig = null;
            }

            // Load the model
            MemorySegment rawPtr = (MemorySegment) LlamaBindings.llama_model_load_from_file.invoke(cPath, modelParams);

            if (rawPtr.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Failed to load model: " + path);
            }

            this.modelPtr = rawPtr.reinterpret(Long.MAX_VALUE);

            // Extract the Vocab pointer immediately
            this.vocabPtr = (MemorySegment) LlamaBindings.llama_model_get_vocab.invoke(modelPtr);

            // Cache metadata - Note: nVocab now uses vocabPtr
            this.nVocab = (int) LlamaBindings.llama_vocab_n_tokens.invoke(vocabPtr);
            this.nCtxTrain = (int) LlamaBindings.llama_model_n_ctx_train.invoke(modelPtr);
            this.nEmbd = (int) LlamaBindings.llama_model_n_embd.invoke(modelPtr);
            this.nLayer = (int) LlamaBindings.llama_model_n_layer.invoke(modelPtr);
            this.nParams = (long) LlamaBindings.llama_model_n_params.invoke(modelPtr);
            this.modelSizeBytes = Files.size(Path.of(modelPath));

        } catch (Throwable t) {
            modelArena.close();
            throw new RuntimeException("Failed to load model", t);
        }
    }

    /**
     * Apply model configuration to model parameters struct
     */
    private void applyModelConfigToModelParams(MemorySegment modelParams, ModelConfig config) {
        try {
            // Set GPU layers
            VarHandle gpuLayersHandle = LlamaBindings.MODEL_PARAMS_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("n_gpu_layers"));
            gpuLayersHandle.set(modelParams, 0L, config.getGpuLayers());

            // Set mmap usage (for SSD offloading)
            VarHandle mmapHandle = LlamaBindings.MODEL_PARAMS_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("use_mmap"));
            mmapHandle.set(modelParams, 0L, (byte) (modelConfig.isUseMmap() ? 1 : 0));

            // Set mlock usage (prevent swap for RAM-resident models)
            VarHandle mlockHandle = LlamaBindings.MODEL_PARAMS_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("use_mlock"));
            mlockHandle.set(modelParams, 0L, (byte) (modelConfig.isUseMlock() ? 1 : 0));

        } catch (Throwable t) {
            throw new RuntimeException("Failed to apply model config to model params", t);
        }
    }

    /**
     * Get the native pointer to the model
     * Used internally for context creation
     */
    public MemorySegment ptr() {
        return modelPtr;
    }

    /**
     * Get model path
     */
    public String getPath() {
        return modelPath;
    }

    /**
     * Get model configuration used for this model
     */
    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    /**
     * Get vocabulary size
     */
    public int getVocabSize() {
        return nVocab;
    }

    public MemorySegment vocabPtr() { return vocabPtr; }

    /**
     * Get training context size
     */
    public int getTrainContextSize() {
        return nCtxTrain;
    }

    /**
     * Get embedding dimensions
     */
    public int getEmbeddingSize() {
        return nEmbd;
    }

    /**
     * Get number of layers
     */
    public int getLayerCount() {
        return nLayer;
    }

    /**
     * Get total parameter count
     */
    public long getParameterCount() {
        return nParams;
    }

    /**
     * Get model size in bytes
     */
    public long getModelSizeBytes() {
        return modelSizeBytes;
    }

    /**
     * Get model size in GB
     */
    public double getModelSizeGB() {
        return modelSizeBytes / 1_000_000_000.0;
    }

    /**
     * Get a human-readable description of the model
     */
    public String getDescription() {
        try (Arena tempArena = Arena.ofConfined()) {
            MemorySegment buffer = tempArena.allocate(256);
            int len = (int) LlamaBindings.llama_model_desc.invoke(modelPtr, buffer, 256L);
            if (len > 0) {
                return buffer.reinterpret(len).getString(0);
            }
            return "Unknown";
        } catch (Throwable t) {
            return "Unknown";
        }
    }

    /**
     * Get BOS (Beginning of Sequence) token ID
     */
    public int getBosToken() {
        try {
            return (int) LlamaBindings.llama_vocab_bos.invoke(vocabPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get BOS token", t);
        }
    }

    /**
     * Get EOS (End of Sequence) token ID
     */
    public int getEosToken() {
        try {
            return (int) LlamaBindings.llama_vocab_eos.invoke(vocabPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get EOS token", t);
        }
    }

    /**
     * Get EOT (End of Turn) token ID
     */
    public int getEotToken() {
        try {
            return (int) LlamaBindings.llama_vocab_eot.invoke(vocabPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get EOT token", t);
        }
    }

    /**
     * Get newline token ID
     */
    public int getNewlineToken() {
        try {
            return (int) LlamaBindings.llama_vocab_nl.invoke(vocabPtr);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get newline token", t);
        }
    }

    /**
     * Retrieves the GGUF chat template (e.g., Jinja2 format) from the model.
     * Returns null if no template is found.
     */
    public String getChatTemplate() {
        try {
            // Pass NULL for the 'name' parameter to get the default template
            MemorySegment templatePtr = (MemorySegment) LlamaBindings.llama_model_chat_template.invoke(modelPtr, MemorySegment.NULL);

            if (templatePtr.equals(MemorySegment.NULL)) {
                return null;
            }

            // Convert the native C-string to a Java String
            return templatePtr.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to retrieve chat template", t);
        }
    }

    /**
     * Tokenize a string into token IDs
     *
     * @param text Text to tokenize
     * @param addBos Whether to add BOS token
     * @param special Whether to parse special tokens
     * @return Array of token IDs
     */
    public int[] tokenize(String text, boolean addBos, boolean special) {
        try (Arena tempArena = Arena.ofConfined()) {
            // Allocate C string
            MemorySegment textSeg = tempArena.allocateFrom(text);
            int textLen = text.length();

            // Estimate max tokens (generous buffer)
            int maxTokens = textLen + 256;
            MemorySegment tokenBuf = tempArena.allocate(ValueLayout.JAVA_INT, maxTokens);

            // Call llama_tokenize with text_len
            int nTokens = (int) LlamaBindings.llama_tokenize.invoke(
                vocabPtr,
                textSeg,
                textLen,
                tokenBuf,
                maxTokens,
                addBos,
                special
            );

            if (nTokens < 0) {
                throw new RuntimeException("Tokenization failed with code: " + nTokens);
            }

            // Copy tokens to Java array
            int[] tokens = new int[nTokens];
            for (int i = 0; i < nTokens; i++) {
                tokens[i] = tokenBuf.getAtIndex(ValueLayout.JAVA_INT, i);
            }
            return tokens;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to tokenize: " + text, t);
        }
    }

    /**
     * Convert a single token ID to its string representation
     */
    public String tokenToString(int tokenId) {
        try (Arena tempArena = Arena.ofConfined()) {
            MemorySegment buffer = tempArena.allocate(256);

            // llama_token_to_piece signature:
            // int32_t llama_token_to_piece(vocab, token, buf, length, lstrip, special)
            int len = (int) LlamaBindings.llama_token_to_piece.invoke(
                vocabPtr,
                tokenId,
                buffer,
                256,     // buffer length
                0,       // lstrip (chars to remove from left)
                true     // special tokens
            );

            if (len < 0) {
                return "";
            }

            byte[] bytes = new byte[len];
            MemorySegment.copy(buffer, ValueLayout.JAVA_BYTE, 0, bytes, 0, len);
            return new String(bytes, StandardCharsets.UTF_8);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to convert token to string: " + tokenId, t);
        }
    }

    /**
     * Detokenize an array of tokens to text
     */
    public String detokenize(int[] tokens, boolean special, boolean unparse) {
        try (Arena tempArena = Arena.ofConfined()) {
            // Allocate token array
            MemorySegment tokenSeg = tempArena.allocate(ValueLayout.JAVA_INT, tokens.length);
            for (int i = 0; i < tokens.length; i++) {
                tokenSeg.setAtIndex(ValueLayout.JAVA_INT, i, tokens[i]);
            }

            // Allocate output buffer
            int bufSize = tokens.length * 8 + 256;
            MemorySegment textBuf = tempArena.allocate(bufSize);

            // Call llama_detokenize
            int len = (int) LlamaBindings.llama_detokenize.invoke(
                vocabPtr,
                tokenSeg,
                tokens.length,
                textBuf,
                bufSize,
                special,
                unparse
            );

            if (len < 0) {
                throw new RuntimeException("Detokenization failed with code: " + len);
            }

            return textBuf.asSlice(0, len).getString(0);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to detokenize", t);
        }
    }

    public void printInfo() {
        System.out.println("=".repeat(60));
        System.out.println("Model Information");
        System.out.println("=".repeat(60));
        System.out.println("Model Name       " + Paths.get(modelPath).getFileName().toString());
        System.out.println("Model Template   " + TemplateDetector.detectTemplate(getChatTemplate()));
        System.out.println("Vocab Size:      " + String.format("%,d", nVocab));
        System.out.println("Embedding Dim:   " + nEmbd);
        System.out.println("Layers:          " + nLayer);
        System.out.println("Train Context:   " + String.format("%,d", nCtxTrain));
        System.out.println("BOS Token:       " + getBosToken());
        System.out.println("EOS Token:       " + getEosToken());
        System.out.println("=".repeat(60));
        System.out.println("\n\n");
    }

    @Override
    public void close() {
        try {
            // Free the model
            LlamaBindings.llama_model_free.invoke(modelPtr);
        } catch (Throwable t) {
            // Log but don't throw - we're in cleanup
            System.err.println("Warning: Failed to free model: " + t.getMessage());
        } finally {
            // Always close the arena
            modelArena.close();
        }
    }
}