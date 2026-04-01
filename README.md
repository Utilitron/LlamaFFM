# LlamaFFM 🦙
High-performance, Zero-JNI Java bindings for llama.cpp

LlamaFFM is a modern Java bridge for the llama.cpp inference engine. 
This project utilizes Project Panama (Foreign Function & Memory API) to provide near-native performance with the safety and ergonomics of modern Java (JDK 22+).


**Key Features:**

Zero JNI Overhead: Direct native calls using the FFM API (java.lang.foreign), bypassing the "JNI tax" and complex C++ glue code.

Modern Memory Management: Leverages java.lang.foreign.Arena for deterministic, safe, and efficient off-heap memory management.

Type-Safe Structs: Full mapping of llama_model, llama_context, and llama_batch to Java MemorySegment layouts.

GPU Accelerated: Built-in support for CUDA, ROCm, and Metal backends via the underlying llama.cpp shared library.

Low-Level Control: Designed for building AI kernels and multi-agent frameworks where VRAM management and inference speed are critical.


**Technical Stack**

Runtime: JDK 22+ (Required for finalized FFM API)

Native Engine: llama.cpp 

Shared Library: You must have the compiled .so (Linux), .dll (Windows), or .dylib (macOS) for your specific architecture.

Validated Build: 8562 (Commit: c46758d28)

Compiler: GCC 13.3.0

Tested OS: Linux (Ubuntu/Mint) with Vulkan backend
