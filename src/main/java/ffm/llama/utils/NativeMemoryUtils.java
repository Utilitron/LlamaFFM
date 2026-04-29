package ffm.llama.utils;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * Safe memory access utilities that eliminate unsafe reinterpret(Long.MAX_VALUE) usage.
 */
public final class NativeMemoryUtils {
    
    private NativeMemoryUtils() {
        throw new AssertionError("Utility class");
    }
    
    /**
     * Safely read a null-terminated C string from native memory.
     * <p>
     * REPLACES: ptr.reinterpret(Long.MAX_VALUE).getString(0)
     *
     * @param ptr Pointer to null-terminated C string
     * @return Java String, or null if ptr is NULL
     */
    public static String readCString(MemorySegment ptr) {
        if (ptr == null || ptr.equals(MemorySegment.NULL) || ptr.address() == 0L) {
            return null;
        }
        
        // Find string length using strnlen-like approach
        long maxScan = 65536; // Safety limit: scan max 64KB
        long len = 0;
        
        // Create a safe bounded view for length detection
        MemorySegment bounded = ptr.reinterpret(maxScan);
        
        while (len < maxScan) {
            byte b = bounded.get(ValueLayout.JAVA_BYTE, len);
            if (b == 0) {
                break;
            }
            len++;
        }
        
        if (len == maxScan) {
            throw new IllegalStateException("String exceeds safety limit (no null terminator found in first 64KB)");
        }
        
        // Now read the exact string with proper bounds
        if (len == 0) {
            return "";
        }
        
        MemorySegment safeView = ptr.reinterpret(len);
        byte[] bytes = new byte[(int) len];
        MemorySegment.copy(safeView, ValueLayout.JAVA_BYTE, 0, bytes, 0, (int) len);
        
        return new String(bytes, StandardCharsets.UTF_8);
    }
    
    /**
     * Safely read a C string with known exact length (not null-terminated).
     * Use when the length is known from a previous native call (e.g., llama_model_desc).
     *
     * @param ptr     Pointer to C string
     * @param byteLen Exact length in bytes
     * @return Java String
     */
    public static String readCStringExact(MemorySegment ptr, long byteLen) {
        if (ptr == null || ptr.equals(MemorySegment.NULL) || ptr.address() == 0L) {
            return null;
        }
        if (byteLen == 0) {
            return "";
        }
        MemorySegment safe = ptr.reinterpret(byteLen);
        byte[] bytes = new byte[(int) byteLen];
        MemorySegment.copy(safe, ValueLayout.JAVA_BYTE, 0, bytes, 0, (int) byteLen);
        return new String(bytes, StandardCharsets.UTF_8);
    }
    
    /**
     * Safely read a C string with known maximum length.
     * More efficient than readCString when length is bounded.
     *
     * @param ptr    Pointer to C string
     * @param maxLen Maximum expected length
     * @return Java String
     */
    public static String readCStringBounded(MemorySegment ptr, int maxLen) {
        if (ptr == null || ptr.equals(MemorySegment.NULL) || ptr.address() == 0L) {
            return null;
        }
        
        MemorySegment bounded = ptr.reinterpret(maxLen);
        
        // Find actual length
        int len = 0;
        while (len < maxLen && bounded.get(ValueLayout.JAVA_BYTE, len) != 0) {
            len++;
        }
        
        if (len == 0) {
            return "";
        }
        
        byte[] bytes = new byte[len];
        MemorySegment.copy(bounded, ValueLayout.JAVA_BYTE, 0, bytes, 0, len);
        
        return new String(bytes, StandardCharsets.UTF_8);
    }
    
    /**
     * Safely create a float array view from native memory.
     * Used for logits and embeddings.
     * <p>
     * REPLACES: ptr.reinterpret(Long.MAX_VALUE) followed by unsafe array access
     *
     * @param ptr          Native pointer to float array
     * @param elementCount Number of float elements
     * @return Safe MemorySegment view with exact bounds
     */
    public static MemorySegment asFloatArray(MemorySegment ptr, long elementCount) {
        if (ptr == null || ptr.equals(MemorySegment.NULL)) {
            throw new IllegalArgumentException("Cannot create array view from NULL pointer");
        }
        
        if (elementCount < 0) {
            throw new IllegalArgumentException("Element count must be non-negative: " + elementCount);
        }
        
        if (elementCount > Integer.MAX_VALUE / Float.BYTES) {
            throw new IllegalArgumentException("Array too large: " + elementCount + " elements");
        }
        
        long byteSize = elementCount * Float.BYTES;
        return ptr.reinterpret(byteSize);
    }
    
    /**
     * Safely create an int array view from native memory.
     * Used for token buffers.
     *
     * @param ptr          Native pointer to int array
     * @param elementCount Number of int elements
     * @return Safe MemorySegment view with exact bounds
     */
    public static MemorySegment asIntArray(MemorySegment ptr, long elementCount) {
        if (ptr == null || ptr.equals(MemorySegment.NULL)) {
            throw new IllegalArgumentException("Cannot create array view from NULL pointer");
        }
        
        if (elementCount < 0) {
            throw new IllegalArgumentException("Element count must be non-negative: " + elementCount);
        }
        
        if (elementCount > Integer.MAX_VALUE / Integer.BYTES) {
            throw new IllegalArgumentException("Array too large: " + elementCount + " elements");
        }
        
        long byteSize = elementCount * Integer.BYTES;
        return ptr.reinterpret(byteSize);
    }
    
    /**
     * Copy float array from native memory to Java array.
     * Ensures bounds safety and prevents buffer overflows.
     *
     * @param ptr   Native pointer to float data
     * @param count Number of floats to copy
     * @return Java float array
     */
    public static float[] copyFloatArray(MemorySegment ptr, int count) {
        if (count == 0) {
            return new float[0];
        }
        
        MemorySegment safe = asFloatArray(ptr, count);
        float[] result = new float[count];
        
        for (int i = 0; i < count; i++) {
            result[i] = safe.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        
        return result;
    }
    
    /**
     * Copy int array from native memory to Java array.
     *
     * @param ptr   Native pointer to int data
     * @param count Number of ints to copy
     * @return Java int array
     */
    public static int[] copyIntArray(MemorySegment ptr, int count) {
        if (count == 0) {
            return new int[0];
        }
        
        MemorySegment safe = asIntArray(ptr, count);
        int[] result = new int[count];
        
        for (int i = 0; i < count; i++) {
            result[i] = safe.getAtIndex(ValueLayout.JAVA_INT, i);
        }
        
        return result;
    }
    
    /**
     * Create a safe opaque handle for native pointers where internal structure is unknown.
     * Returns a zero-length segment that can be passed to native functions but prevents
     * accidental memory access.
     * <p>
     * Use this for: model pointers, context pointers, sampler pointers
     *
     * @param ptr Raw pointer from native code
     * @return Safe zero-length handle
     */
    public static MemorySegment asOpaqueHandle(MemorySegment ptr) {
        if (ptr == null || ptr.equals(MemorySegment.NULL)) {
            return MemorySegment.NULL;
        }
        
        // Keep it zero-length - we never dereference opaque handles
        return ptr.reinterpret(0);
    }
    
    /**
     * Verify that a pointer is valid (non-null, non-zero).
     *
     * @param ptr  Pointer to check
     * @param name Descriptive name for error messages
     * @throws IllegalStateException if pointer is invalid
     */
    public static void requireValid(MemorySegment ptr, String name) {
        if (ptr == null || ptr.equals(MemorySegment.NULL) || ptr.address() == 0L) {
            throw new IllegalStateException(name + " pointer is invalid (null or zero)");
        }
    }
}