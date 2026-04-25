package ffm.llama.message;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MessageRoleTest {

    @Nested
    @DisplayName("fromString parsing")
    class FromString {

        @Test
        @DisplayName("Should parse valid roles case‑insensitively")
        void shouldParseValidRoles() {
            assertEquals(MessageRole.SYSTEM, MessageRole.fromString("system"));
            assertEquals(MessageRole.USER, MessageRole.fromString("User"));
            assertEquals(MessageRole.ASSISTANT, MessageRole.fromString("ASSISTANT"));
            assertEquals(MessageRole.TOOL, MessageRole.fromString("tOOl"));
        }

        @Test
        @DisplayName("Should throw for null input")
        void shouldThrowForNull() {
            assertThrows(IllegalArgumentException.class,
                    () -> MessageRole.fromString(null));
        }

        @Test
        @DisplayName("Should throw with descriptive message for unknown roles")
        void shouldThrowForUnknownRole() {
            Exception ex = assertThrows(IllegalArgumentException.class,
                    () -> MessageRole.fromString("admin"));
            assertTrue(ex.getMessage().contains("Unknown message role"));
            assertTrue(ex.getMessage().contains("Valid roles are"));
        }
    }

    @Test
    @DisplayName("getValue should return the correct string")
    void getValueShouldReturnCorrectString() {
        assertEquals("system", MessageRole.SYSTEM.getValue());
        assertEquals("user", MessageRole.USER.getValue());
        assertEquals("assistant", MessageRole.ASSISTANT.getValue());
        assertEquals("tool", MessageRole.TOOL.getValue());
    }

    @Test
    @DisplayName("toString should be same as getValue")
    void toStringShouldReturnValue() {
        assertEquals(MessageRole.SYSTEM.getValue(), MessageRole.SYSTEM.toString());
    }
}
