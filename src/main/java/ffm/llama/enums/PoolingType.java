package ffm.llama.enums;

/**
 * Pooling Types
 */
public enum PoolingType {
    UNSPECIFIED(-1),
    NONE(0),
    MEAN(1),
    CLS(2),
    LAST(3),
    RANK(4);
    
    private final int value;
    
    PoolingType(int value) {
        this.value = value;
    }
    
    public static PoolingType fromValue(int value) {
        for (PoolingType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        return UNSPECIFIED;
    }
    
    public int getValue() {
        return value;
    }
}
