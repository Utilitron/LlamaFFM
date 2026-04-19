# LlamaFFM 🦙
High-performance, Zero-JNI Java bindings for llama.cpp

LlamaFFM is a modern Java bridge for the llama.cpp inference engine. 
This project utilizes Project Panama (Foreign Function & Memory API) to provide near-native performance with the safety and ergonomics of modern Java (JDK 22+).


## Key Features:
- Zero JNI Overhead: Direct native calls using the FFM API (java.lang.foreign), bypassing the "JNI tax" and complex C++ glue code.
- Modern Memory Management: Leverages java.lang.foreign.Arena for deterministic, safe, and efficient off-heap memory management.
- Type-Safe Structs: Full mapping of llama_model, llama_context, and llama_batch to Java MemorySegment layouts.
- GPU Accelerated: Built-in support for CUDA, ROCm, and Metal backends via the underlying llama.cpp shared library.
- Low-Level Control: Designed for building AI kernels and multi-agent frameworks where VRAM management and inference speed are critical.

---

## Installation

### Prerequisites
- JDK 22 or later
- Compiled llama.cpp shared library for your platform:
  - Linux: `libllama.so`
  - macOS: `libllama.dylib`
  - Windows: `llama.dll`
- (Optional) CUDA/ROCm/Metal drivers for GPU acceleration

**Validated Build**: 8562 (Commit: c46758d28)

**Compiler**: GCC 13.3.0

**Tested OS**: Linux (Ubuntu/Mint) with Vulkan backend


### Building the Library
LlamaFFM is not yet published to Maven Central. You must build and install it locally:

```bash
git clone https://github.com/your-org/LlamaFFM.git
cd LlamaFFM
mvn clean install
```

### Maven Dependency
```xml
<dependency>
    <groupId>ffm.llama</groupId>
    <artifactId>LlamaFFM</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

## Native Library Loading

LlamaFFM requires the compiled llama.cpp shared library for your platform:
- **Linux:** `libllama.so`
- **macOS:** `libllama.dylib`
- **Windows:** `llama.dll`

The library is loaded at startup. You can provide it in one of three ways:

### Option 1: System Library Path
Place the shared library in a directory included in your system's library search path:
- **Linux/macOS:** `LD_LIBRARY_PATH` or standard locations (`/usr/lib`, `/usr/local/lib`)
- **Windows:** `PATH` environment variable or system directories

### Option 2: Environment Variable
Set the `LLAMA_LIB_PATH` environment variable to the absolute path of the shared library:
```bash
export LLAMA_LIB_PATH=/path/to/libllama.so

### **Quick Start Example**

A minimal working example builds confidence.

## Quick Start

```java
import ffm.llama.config.ModelConfig;
import ffm.llama.message.ChatMessage;
import ffm.llama.message.MessageRole;
import ffm.llama.sampling.LlamaSampler;
import ffm.llama.service.LlmService;

public class QuickStart {
    public static void main(String[] args) {
        try (LlmService service = new LlmService()) {
            // Load model with GPU offloading
            ModelConfig config = ModelConfig.Builder.create()
                .gpuLayers(99)
                .contextSize(4096)
                .batchSize(512)
                .build();
            
            service.loadModel("/path/to/model.gguf", config);
            
            var conversation = List.of(
                new ChatMessage(MessageRole.SYSTEM, "You are a helpful assistant."),
                new ChatMessage(MessageRole.USER, "Explain Project Panama in one sentence.")
            );
            
            String response = service.generate("model.gguf", conversation);
            System.out.println(response);
        }
    }
}
```

---

## Configuration

### ModelConfig Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `gpuLayers` | Layers to offload to GPU (99 = all) | 0 |
| `offloadKvToGpu` | Store KV cache on GPU | false |
| `useMmap` | Memory‑map model file | true |
| `useMlock` | Lock model in RAM | false |
| `contextSize` | Maximum token context | 2048 |
| `batchSize` | Tokens per decode batch | 512 |
| `cpuThreads` | Threads for CPU inference | #cores |
| `flashAttention` | Enable Flash Attention | false |
| `embeddings` | Enable embedding mode | false |
| `cacheTypeK` / `cacheTypeV` | KV cache quantization | `F16` |

### Sampling

**Predefined configurations:**
- `greedy()` – Deterministic, picks highest probability token.
- `balanced()` – Temperature 0.7, Top‑P 0.9, Top‑K 40.
- `creative()` – Temperature 0.9, Top‑P 0.95.
- `precise()` – Temperature 0.3, Top‑K 10.

**Custom sampler:**
```java
LlamaSampler.SamplerConfig custom = LlamaSampler.SamplerConfig.builder()
    .temperature(0.8f)
    .topP(0.95f)
    .topK(50)
    .build();
```

---

## License

LlamaFFM is licensed under the [MIT License](LICENSE).  
llama.cpp is licensed under the [MIT License](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE).

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.  
Focus areas:
- Expanding the binding coverage of llama.cpp APIs.
- Adding support for more backends (Vulkan, SYCL).
- Improving documentation and examples.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) – The incredible C++ inference engine.
- [Project Panama](https://openjdk.org/projects/panama/) – For native interop without JNI.
