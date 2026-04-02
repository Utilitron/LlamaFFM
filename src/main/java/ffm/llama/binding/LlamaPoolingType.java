package ffm.llama.binding;

public enum LlamaPoolingType {
    UNSPECIFIED(-1),
    NONE(0),
    MEAN(1),
    CLS(2),
    LAST(3),
    RANK(4);

    private final int value;

    LlamaPoolingType(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static LlamaPoolingType fromValue(int value) {
        for (LlamaPoolingType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        return UNSPECIFIED;
    }
}
