package ffm.llama.utils;

import org.junit.jupiter.api.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

class NativeMemoryUtilsTest {
    
    private static final String HELLO = "Hello";
    private static final byte[] HELLO_BYTES = HELLO.getBytes(StandardCharsets.UTF_8);
    
    private Arena arena;
    
    @BeforeEach
    void setUp() {
        arena = Arena.ofConfined();
    }
    
    @AfterEach
    void tearDown() {
        arena.close();
    }
    
    @Nested
    @DisplayName("readCStringExact")
    class ReadCStringExact {
        
        @Test
        @DisplayName("Should return null for NULL pointer")
        void shouldReturnNullForNullPointer() {
            assertNull(NativeMemoryUtils.readCStringExact(MemorySegment.NULL, 0));
        }
        
        @Test
        @DisplayName("Should return empty string for zero length")
        void shouldReturnEmptyStringForZeroLength() {
            MemorySegment seg = arena.allocateFrom(HELLO);
            assertEquals("", NativeMemoryUtils.readCStringExact(seg, 0));
        }
        
        @Test
        @DisplayName("Should read exact number of bytes without null-term requirement")
        void shouldReadExactBytes() {
            MemorySegment seg = arena.allocateFrom(HELLO);
            String result = NativeMemoryUtils.readCStringExact(seg, HELLO.length());
            assertEquals(HELLO, result);
        }
        
        @Test
        @DisplayName("Should read bytes even if they contain embedded zeros")
        void shouldReadBytesWithEmbeddedNull() {
            byte[] data = {'A', 0, 'B'};
            MemorySegment seg = arena.allocate(ValueLayout.JAVA_BYTE, data.length);
            for (int i = 0; i < data.length; i++) {
                seg.set(ValueLayout.JAVA_BYTE, i, data[i]);
            }
            String result = NativeMemoryUtils.readCStringExact(seg, data.length);
            assertEquals(3, result.length());
            assertEquals('A', result.charAt(0));
            assertEquals(0, result.charAt(1));    // null char
            assertEquals('B', result.charAt(2));
        }
    }
    
    @Nested
    @DisplayName("readCStringBounded")
    class ReadCStringBounded {
        
        @Test
        @DisplayName("Should return null for NULL pointer")
        void shouldReturnNullForNullPointer() {
            assertNull(NativeMemoryUtils.readCStringBounded(MemorySegment.NULL, 10));
        }
        
        @Test
        @DisplayName("Should return empty string when first byte is null")
        void shouldReturnEmptyStringWhenFirstByteIsNull() {
            MemorySegment seg = arena.allocate(1);
            seg.set(ValueLayout.JAVA_BYTE, 0, (byte) 0);
            assertEquals("", NativeMemoryUtils.readCStringBounded(seg, 10));
        }
        
        @Test
        @DisplayName("Should read until null within bound")
        void shouldReadUntilNullWithinBound() {
            String text = "Test\0trailing";
            MemorySegment seg = arena.allocateFrom(text);
            String result = NativeMemoryUtils.readCStringBounded(seg, 20);
            assertEquals("Test", result);
        }
        
        @Test
        @DisplayName("Should read up to maxLen if no null found")
        void shouldReadUpToMaxLenWhenNoNull() {
            MemorySegment seg = arena.allocate(5);
            for (int i = 0; i < 5; i++) {
                seg.set(ValueLayout.JAVA_BYTE, i, (byte) 'A');
            }
            String result = NativeMemoryUtils.readCStringBounded(seg, 5);
            assertEquals("AAAAA", result);
        }
    }
    
    @Nested
    @DisplayName("readCString")
    class ReadCString {
        
        @Test
        @DisplayName("Should return null for NULL pointer")
        void shouldReturnNullForNullPointer() {
            assertNull(NativeMemoryUtils.readCString(MemorySegment.NULL));
        }
        
        @Test
        @DisplayName("Should return empty string for immediate null terminator")
        void shouldReturnEmptyStringForImmediateNull() {
            MemorySegment seg = arena.allocate(1);
            seg.set(ValueLayout.JAVA_BYTE, 0, (byte) 0);
            assertEquals("", NativeMemoryUtils.readCString(seg));
        }
        
        @Test
        @DisplayName("Should read a short null-terminated string")
        void shouldReadShortNullTerminatedString() {
            String text = "Hello\0";
            MemorySegment seg = arena.allocateFrom(text);
            assertEquals("Hello", NativeMemoryUtils.readCString(seg));
        }
        
        @Test
        @DisplayName("Should throw if no null terminator within 64KB")
        void shouldThrowIfNoNullWithin64K() {
            MemorySegment seg = arena.allocate(65537);
            for (int i = 0; i < 65537; i++) {
                seg.set(ValueLayout.JAVA_BYTE, i, (byte) 'x');
            }
            assertThrows(IllegalStateException.class,
                    () -> NativeMemoryUtils.readCString(seg));
        }
    }
    
    @Nested
    @DisplayName("asFloatArray / asIntArray")
    class ArrayViews {
        
        @Test
        @DisplayName("asFloatArray should reject NULL pointer")
        void shouldRejectNullFloat() {
            assertThrows(IllegalArgumentException.class,
                    () -> NativeMemoryUtils.asFloatArray(MemorySegment.NULL, 10));
        }
        
        @Test
        @DisplayName("asFloatArray should create bounded view")
        void shouldCreateFloatView() {
            MemorySegment seg = arena.allocate(20 * 4);
            MemorySegment view = NativeMemoryUtils.asFloatArray(seg, 20);
            assertEquals(20 * 4, view.byteSize());
        }
        
        @Test
        @DisplayName("asIntArray should create bounded view")
        void shouldCreateIntView() {
            MemorySegment seg = arena.allocate(10 * 4);
            MemorySegment view = NativeMemoryUtils.asIntArray(seg, 10);
            assertEquals(10 * 4, view.byteSize());
        }
    }
    
    @Nested
    @DisplayName("copyFloatArray / copyIntArray")
    class CopyArrays {
        
        @Test
        @DisplayName("copyFloatArray should copy exact number of floats")
        void shouldCopyFloats() {
            int count = 5;
            MemorySegment seg = arena.allocate(count * 4);
            for (int i = 0; i < count; i++) {
                seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) i * 1.5f);
            }
            float[] result = NativeMemoryUtils.copyFloatArray(seg, count);
            assertEquals(count, result.length);
            for (int i = 0; i < count; i++) {
                assertEquals(i * 1.5f, result[i], 0.0);
            }
        }
        
        @Test
        @DisplayName("copyIntArray should copy exact number of ints")
        void shouldCopyInts() {
            int count = 3;
            MemorySegment seg = arena.allocate(count * 4);
            seg.setAtIndex(ValueLayout.JAVA_INT, 0, 100);
            seg.setAtIndex(ValueLayout.JAVA_INT, 1, 200);
            seg.setAtIndex(ValueLayout.JAVA_INT, 2, 300);
            int[] result = NativeMemoryUtils.copyIntArray(seg, count);
            assertArrayEquals(new int[]{100, 200, 300}, result);
        }
        
        @Test
        @DisplayName("copyFloatArray should return empty array for zero count")
        void shouldReturnEmptyForZeroCountFloat() {
            float[] result = NativeMemoryUtils.copyFloatArray(MemorySegment.NULL, 0);
            assertEquals(0, result.length);
        }
    }
    
    @Nested
    @DisplayName("asOpaqueHandle")
    class OpaqueHandle {
        
        @Test
        @DisplayName("Should return NULL for NULL input")
        void shouldReturnNullForNull() {
            assertSame(MemorySegment.NULL, NativeMemoryUtils.asOpaqueHandle(MemorySegment.NULL));
        }
        
        @Test
        @DisplayName("Should return a zero-length view for valid pointer")
        void shouldReturnZeroLength() {
            MemorySegment seg = arena.allocate(100);
            MemorySegment handle = NativeMemoryUtils.asOpaqueHandle(seg);
            assertEquals(0, handle.byteSize());
            assertEquals(seg.address(), handle.address());
        }
    }
    
    @Nested
    @DisplayName("requireValid")
    class RequireValid {
        
        @Test
        @DisplayName("Should throw for NULL pointer")
        void shouldThrowForNull() {
            assertThrows(IllegalStateException.class,
                    () -> NativeMemoryUtils.requireValid(MemorySegment.NULL, "test"));
        }
        
        @Test
        @DisplayName("Should not throw for valid pointer")
        void shouldNotThrowForValid() {
            MemorySegment seg = arena.allocate(1);
            assertDoesNotThrow(() -> NativeMemoryUtils.requireValid(seg, "test"));
        }
    }
}
