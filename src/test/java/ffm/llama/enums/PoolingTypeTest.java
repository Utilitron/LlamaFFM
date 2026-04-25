package ffm.llama.enums;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link PoolingType}.
 * Focuses on the {@code fromValue} lookup contract.
 */
class PoolingTypeTest {

    @Nested
    @DisplayName("fromValue lookup")
    class FromValueLookup {

        @Test
        @DisplayName("Should map known values to correct enum constant")
        void shouldMapKnownValues() {
            assertEquals(PoolingType.NONE, PoolingType.fromValue(0));
            assertEquals(PoolingType.MEAN, PoolingType.fromValue(1));
            assertEquals(PoolingType.CLS, PoolingType.fromValue(2));
            assertEquals(PoolingType.LAST, PoolingType.fromValue(3));
            assertEquals(PoolingType.RANK, PoolingType.fromValue(4));
            assertEquals(PoolingType.UNSPECIFIED, PoolingType.fromValue(-1));
        }

        @Test
        @DisplayName("Should return UNSPECIFIED for unknown values")
        void shouldReturnUnspecifiedForUnknown() {
            assertEquals(PoolingType.UNSPECIFIED, PoolingType.fromValue(5));
            assertEquals(PoolingType.UNSPECIFIED, PoolingType.fromValue(Integer.MIN_VALUE));
            assertEquals(PoolingType.UNSPECIFIED, PoolingType.fromValue(Integer.MAX_VALUE));
        }

        @Test
        @DisplayName("Should return UNSPECIFIED for negative values except -1")
        void shouldReturnUnspecifiedForNegativeValues() {
            assertEquals(PoolingType.UNSPECIFIED, PoolingType.fromValue(-2));
            assertEquals(PoolingType.UNSPECIFIED, PoolingType.fromValue(-100));
        }
    }

    @Nested
    @DisplayName("value getter")
    class ValueGetter {

        @Test
        @DisplayName("Should return the correct int value for each constant")
        void shouldReturnCorrectValue() {
            assertEquals(-1, PoolingType.UNSPECIFIED.getValue());
            assertEquals(0, PoolingType.NONE.getValue());
            assertEquals(1, PoolingType.MEAN.getValue());
            assertEquals(2, PoolingType.CLS.getValue());
            assertEquals(3, PoolingType.LAST.getValue());
            assertEquals(4, PoolingType.RANK.getValue());
        }
    }
}